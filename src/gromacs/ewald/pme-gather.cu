#include "pme.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include <cuda.h>

#include "pme-timings.cuh"

#include "pme-internal.h"
#include "pme-cuda.cuh"

#include <assert.h>

void pme_gpu_alloc_gather_forces(gmx_pme_t *pme)
{
    const int n = pme->atc[0].n; //?
    assert(n > 0);
    const int forcesSize = DIM * n * sizeof(real);
    pme->gpu->forces = (real *)PMEMemoryFetch(pme, PME_ID_FORCES, forcesSize, ML_DEVICE);
}

void pme_gpu_get_forces(gmx_pme_t *pme)
{
    cudaStream_t s = pme->gpu->pmeStream;
    cudaError_t stat = cudaStreamWaitEvent(s, pme->gpu->syncForcesD2H, 0);
    CU_RET_ERR(stat, "error while waiting for PME forces");

    const int n = pme->atc[0].n;
    const int forcesSize = DIM * n * sizeof(real);
    real *forces = (real *)PMEMemoryFetch(pme, PME_ID_FORCES, forcesSize, ML_HOST);
    memcpy(pme->atc[0].f, forces, forcesSize);
    // did not succeed in using cudaHostRegister instead of memcpy

    //pme_gpu_sloppy_force_reduction(pme, forces);
    // done on the GPU instead by passing bOverwriteForces = FALSE to the gather functions
}

void pme_gpu_copy_forces(gmx_pme_t *pme)
{
    // host-to-device copy of the forces to be reduces with the gather results
    // we have to be sure that the atc forces (listed and such) are already calculated

    const int n = pme->atc[0].n;
    assert(n);
    const int forcesSize = DIM * n * sizeof(real);
    real *forces = (real *)PMEMemoryFetch(pme, PME_ID_FORCES, forcesSize, ML_HOST);
    memcpy(forces, pme->atc[0].f, forcesSize);
    cu_copy_H2D_async(pme->gpu->forces, forces, forcesSize, pme->gpu->pmeStream);
}

template <
        const int order,
        const int particlesPerBlock,
        const gmx_bool bOverwriteForces
        >
__launch_bounds__(4 * warp_size, 16)
__global__ void pme_gather_kernel
(const real * __restrict__ gridGlobal, const int n,
 const pme_gpu_const_parameters constants, const int pnx, const int pny, const int pnz,
 const real * __restrict__ thetaGlobal,
 const real * __restrict__ dthetaGlobal,
 real * __restrict__ forcesGlobal, const real * __restrict__ coefficientGlobal,
 #if !PME_EXTERN_CMEM
  const struct pme_gpu_recipbox_t RECIPBOX,
 #endif
 const int * __restrict__ idxGlobal
 )
{
    /* sum forces for local particles */

    // these are particle indices - in shared and global memory
    const int localIndex = threadIdx.z;
    const int globalIndex = blockIdx.x * blockDim.z + threadIdx.z;

    const int particleDataSize = order * order;
    const int blockSize = particlesPerBlock * particleDataSize; //1 line per thread
    // should the array size aligned by warp size for shuffle?

    const int thetaSize = PME_SPREADGATHER_BLOCK_DATA_SIZE * order;
    const int idxSize = PME_SPREADGATHER_BLOCK_DATA_SIZE;
    __shared__ int idx[idxSize];
    __shared__ real theta[thetaSize];
    __shared__ real dtheta[thetaSize];


    // spline Y/Z coordinates
    const int ithy = threadIdx.y;
    const int ithz = threadIdx.x;
    // these are spline contribution indices in shared memory
    const int splineIndex = threadIdx.y * blockDim.x + threadIdx.x;   // relative to the current particle
    const int lineIndex = (threadIdx.z * (blockDim.x * blockDim.y)) + splineIndex; // and to all the block's particles


    int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;

    if (threadLocalId < idxSize)
    {
        idx[threadLocalId] = idxGlobal[blockIdx.x * idxSize + threadLocalId];
    }
    if ((threadLocalId < thetaSize))
    {
        theta[threadLocalId] = thetaGlobal[blockIdx.x * thetaSize + threadLocalId];
        dtheta[threadLocalId] = dthetaGlobal[blockIdx.x * thetaSize + threadLocalId];
    }

    //locality?
    __syncthreads();

    real fx = 0.0f;
    real fy = 0.0f;
    real fz = 0.0f;

    if (globalIndex < n)
    {
        const int thetaOffsetBase = localIndex * PME_SPLINE_PARTICLE_STRIDE;
        const int thetaOffsetY = thetaOffsetBase + ithy * PME_SPLINE_ORDER_STRIDE + YY;
        const real ty = theta[thetaOffsetY];
        const real dy = dtheta[thetaOffsetY];
        const int thetaOffsetZ = thetaOffsetBase + ithz * PME_SPLINE_ORDER_STRIDE + ZZ;
        const real dz = dtheta[thetaOffsetZ];
        const real tz = theta[thetaOffsetZ];
        const int indexBaseYZ = ((idx[localIndex * DIM + XX] + 0) * pny + (idx[localIndex * DIM + YY] + ithy)) * pnz + (idx[localIndex * DIM + ZZ] + ithz);
#pragma unroll
        for (int ithx = 0; (ithx < order); ithx++)
        {
            const real gridValue = gridGlobal[indexBaseYZ + ithx * pny * pnz];
            const int thetaOffsetX = thetaOffsetBase + ithx * PME_SPLINE_ORDER_STRIDE + XX;
            const real tx = theta[thetaOffsetX];
            const real dx = dtheta[thetaOffsetX];
            const real fxy1 = tz * gridValue;
            const real fz1  = dz * gridValue;
            fx += dx * ty * fxy1;
            fy += tx * dy * fxy1;
            fz += tx * ty * fz1;
            /*
            atomicAdd(fx + localIndex, dx * ty * fxy1);
            atomicAdd(fy + localIndex, tx * dy * fxy1);
            atomicAdd(fz + localIndex, tx * ty * fz1);
            */
        }
    }
    __syncthreads();

    // now particlesPerBlock particles have to reduce order^2 contributions each

    __shared__ float3 fSumArray[particlesPerBlock];

#if (GMX_PTX_ARCH >= 300)
    if (!(order & (order - 1))) // only for orders of power of 2
    {
        // a tricky shuffle reduction inspired by reduce_force_j_warp_shfl

        assert(order == 4); // confused about others and the best data layout so far :(
        assert(particleDataSize <= warp_size);
        const int width = particleDataSize;
        // have to rework for particleDataSize > warp_size (order 8 or larger...)

        fx += __shfl_down(fx, 1, width);
        fy += __shfl_up  (fy, 1, width);
        fz += __shfl_down(fz, 1, width);

        if (splineIndex & 1)
        {
            fx = fy;
        }

        fx += __shfl_down(fx, 2, width);
        fz += __shfl_up  (fz, 2, width);

        if (splineIndex & 2)
        {
            fx = fz;
        }

        // by now fx contains intermediate sums of all 3 components in groups of 4:
        // splineIndex    0            1            2 and 3      4            5            6 and 7      8...
        // sum of...      fx0 to fx3   fy0 to fy3   fz0 to fz3   fx4 to fx7   fy4 to fy7   fz4 to fz7   etc.

        // we have to just further reduce those groups of 4
        for (int delta = 4; delta < particleDataSize; delta <<= 1)
        {
            fx += __shfl_down(fx, delta, width);
        }

        if (splineIndex < 3)
            *((real *)(&fSumArray[localIndex]) + splineIndex) = fx * constants.nXYZ[splineIndex];
    }
    else
#endif
    {
        // lazy 3-thread reduction in shared memory inspired by reduce_force_j_generic
        __shared__ real fSharedArray[DIM * blockSize];
        fSharedArray[lineIndex] = fx;
        fSharedArray[lineIndex + blockSize] = fy;
        fSharedArray[lineIndex + 2 * blockSize] = fz;

        if (splineIndex < 3)
        {
            float f = 0.0f;
            for (int j = localIndex * particleDataSize; j < (localIndex + 1) * particleDataSize; j++)
            {
                f += fSharedArray[blockSize * splineIndex + j];
            }
            *((real *)(&fSumArray[localIndex]) + splineIndex) = f * constants.nXYZ[splineIndex];
        }
    }
    __syncthreads();

    //reduce by components, again
    if (threadLocalId < DIM * particlesPerBlock)
    {
        // new, different particle indices
        const int localIndexFinal = threadLocalId / DIM;
        const int dimIndex = threadLocalId - localIndexFinal * DIM;

        const float3 fSum = fSumArray[localIndexFinal];
        const int globalIndexFinal = blockIdx.x * particlesPerBlock + localIndexFinal;
        const real coefficient = coefficientGlobal[globalIndexFinal];

        real contrib;
        switch (dimIndex)
        {
            case XX:
            contrib = RECIPBOX.box[XX].x * fSum.x /*+ RECIPBOX.box[YY].x * fSum.y + RECIPBOX.box[ZZ].x * fSum.z*/;
            break;

            case YY:
            contrib = RECIPBOX.box[XX].y * fSum.x + RECIPBOX.box[YY].y * fSum.y /* + RECIPBOX.box[ZZ].y * fSum.z*/;
            break;

            case ZZ:
            contrib = RECIPBOX.box[XX].z * fSum.x + RECIPBOX.box[YY].z * fSum.y + RECIPBOX.box[ZZ].z * fSum.z;
            break;
        }
        contrib *= -coefficient;

        if (bOverwriteForces)
            forcesGlobal[blockIdx.x * PME_SPREADGATHER_BLOCK_DATA_SIZE + threadLocalId] = contrib;
        else
            forcesGlobal[blockIdx.x * PME_SPREADGATHER_BLOCK_DATA_SIZE + threadLocalId] += contrib;
    }
}


// a quick dirty copy of pme_wrap_kernel
template <
    const int order
    >
__global__ void pme_unwrap_kernel
    (const int nx, const int ny, const int nz,
     const int pny, const int pnz,
 #if !PME_EXTERN_CMEM
     const struct pme_gpu_overlap_t OVERLAP,
 #endif
     real * __restrict__ grid
     )
{
    // UNWRAP
    int blockId = blockIdx.x
                 + blockIdx.y * gridDim.x
                 + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                  + (threadIdx.z * (blockDim.x * blockDim.y))
                  + (threadIdx.y * blockDim.x)
                  + threadIdx.x;

    //should use ldg.128

    if (threadId < OVERLAP.overlapCellCounts[OVERLAP_ZONES - 1])
    {
        int zoneIndex = -1;
        do
        {
            zoneIndex++;
        }
        while (threadId >= OVERLAP.overlapCellCounts[zoneIndex]);
        const int2 zoneSizeYZ = OVERLAP.overlapSizes[zoneIndex];
        // this is the overlapped cells's index relative to the current zone
        const int cellIndex = (zoneIndex > 0) ? (threadId - OVERLAP.overlapCellCounts[zoneIndex - 1]) : threadId;

        // replace integer division/modular arithmetics - a big performance hit
        // try int_fastdiv?
        const int ixy = cellIndex / zoneSizeYZ.y; //yupinov check expensive integer divisions everywhere!
        const int iz = cellIndex - zoneSizeYZ.y * ixy;
        const int ix = ixy / zoneSizeYZ.x;
        const int iy = ixy - zoneSizeYZ.x * ix;
        const int sourceIndex = (ix * pny + iy) * pnz + iz;

        int targetOffset = 0;

        // stage those bits in constant memory as well
        const int overlapZ = ((zoneIndex == 0) || (zoneIndex == 3) || (zoneIndex == 4) || (zoneIndex == 6)) ? 1 : 0;
        const int overlapY = ((zoneIndex == 1) || (zoneIndex == 3) || (zoneIndex == 5) || (zoneIndex == 6)) ? 1 : 0;
        const int overlapX = ((zoneIndex == 2) || (zoneIndex > 3)) ? 1 : 0;
        if (overlapZ)
        {
            targetOffset = nz;
        }
        if (overlapY)
        {
            targetOffset += ny * pnz;
        }
        if (overlapX)
        {
            targetOffset += nx * pny * pnz;
        }
        const int targetIndex = sourceIndex + targetOffset;
        grid[targetIndex] = grid[sourceIndex];
    }
}

void gather_f_bsplines_gpu(struct gmx_pme_t *pme, real *grid,
                   pme_atomcomm_t *atc,
                   splinedata_t *spline,
                   real scale,
                   const gmx_bool bOverwriteForces)
{
    int n = spline->n;
    if (!n)
        return;

    if (!bOverwriteForces)
        pme_gpu_copy_forces(pme);

    // false: we use some other GPU forces buffer for the final reduction, so we want to add to that
    // in that case, maybe we want to replace + with atomicAdd at the end of kernel?
    // true: we have our dedicated buffer, so just overwrite directly

    cudaStream_t s = pme->gpu->pmeStream;

    //pme_atomcomm_t atc = pme->atc[0];
    real *atc_coefficient = atc->coefficient;

    const int order = pme->pme_order;

    /*
    const int pnx = pmegrid->n[XX];
    const int pny = pmegrid->n[YY];
    const int pnz = pmegrid->n[ZZ];
    */
    const int pnx   = pme->pmegrid_nx;
    const int pny   = pme->pmegrid_ny;
    const int pnz   = pme->pmegrid_nz;
    const int nx = pme->nkx;
    const int ny = pme->nky;
    const int nz = pme->nkz;

    const int ndatatot = pnx * pny * pnz;
    const int gridSize = ndatatot * sizeof(real);
    if (!pme->gpu->bGPUFFT)
        cu_copy_H2D_async(pme->gpu->grid, grid, gridSize, s);

    if (pme->gpu->bGPUSingle)
    {
        if (order == 4)
        {
            const int blockSize = 4 * warp_size; //yupinov thsi is everywhere! and architecture-specific
            const int overlap = order - 1;

            const int overlappedCells = (nx + overlap) * (ny + overlap) * (nz + overlap) - nx * ny * nz;
            const int nBlocks = (overlappedCells + blockSize - 1) / blockSize;

            pme_gpu_timing_start(pme, ewcsPME_UNWRAP);

            pme_unwrap_kernel<4> <<<nBlocks, blockSize, 0, s>>>(nx, ny, nz, pny, pnz,
#if !PME_EXTERN_CMEM
                                                                pme->gpu->overlap,
#endif
                                                                pme->gpu->grid);

            CU_LAUNCH_ERR("pme_unwrap_kernel");

            pme_gpu_timing_stop(pme, ewcsPME_UNWRAP);

        }
        else
            gmx_fatal(FARGS, "gather: orders other than 4 untested!");
    }

    int forcesSize = DIM * n * sizeof(real);
    int size_indices = n * sizeof(int);
    int size_splines = order * n * sizeof(int);

    real *atc_f_h = (real *)PMEMemoryFetch(pme, PME_ID_FORCES, forcesSize, ML_HOST);

    for (int i = 0; i < n; i++)
    {
        // coefficients
        atc_coefficient[i] *= scale;
    }

    // thetas
    real *theta_d = (real *)PMEMemoryFetch(pme, PME_ID_THETA, DIM * size_splines, ML_DEVICE);
    real *dtheta_d = (real *)PMEMemoryFetch(pme, PME_ID_DTHETA, DIM * size_splines, ML_DEVICE);

    // indices
    int *idx_d = (int *)PMEMemoryFetch(pme, PME_ID_IDXPTR, DIM * size_indices, ML_DEVICE);

    const float3 nXYZ = {(real)nx, (real)ny, (real)nz};
    memcpy(pme->gpu->constants.nXYZ, &nXYZ, sizeof(nXYZ));


    const int blockSize = 4 * warp_size;
    const int particlesPerBlock = blockSize / order / order;
    dim3 nBlocks((n + blockSize - 1) / blockSize * order * order);
    dim3 dimBlock(order, order, particlesPerBlock);

    pme_gpu_timing_start(pme, ewcsPME_GATHER);

    if (order == 4)
        if (bOverwriteForces)
            pme_gather_kernel<4, blockSize / 4 / 4, TRUE> <<<nBlocks, dimBlock, 0, s>>>
              (pme->gpu->grid,
               n,
               pme->gpu->constants, pnx, pny, pnz,
               theta_d, dtheta_d,
               pme->gpu->forces, pme->gpu->coefficients,
#if !PME_EXTERN_CMEM
               pme->gpu->recipbox,
#endif
               idx_d);
        else
            pme_gather_kernel<4, blockSize / 4 / 4, FALSE> <<<nBlocks, dimBlock, 0, s>>>
              (pme->gpu->grid,
               n,
               pme->gpu->constants, pnx, pny, pnz,
               theta_d, dtheta_d,
               pme->gpu->forces, pme->gpu->coefficients,
#if !PME_EXTERN_CMEM
               pme->gpu->recipbox,
#endif
               idx_d);
    else
        gmx_fatal(FARGS, "gather: orders other than 4 untested!");
    CU_LAUNCH_ERR("pme_gather_kernel");

    pme_gpu_timing_stop(pme, ewcsPME_GATHER);

    cu_copy_D2H_async(atc_f_h, pme->gpu->forces, forcesSize, s);
    cudaError_t stat = cudaEventRecord(pme->gpu->syncForcesD2H, s);
    CU_RET_ERR(stat, "PME gather forces sync fail");
}

