#include <assert.h>

#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "pme-cuda.cuh"

// allocate the GPU output buffer for the resulting PME forces
void pme_gpu_alloc_gather_forces(gmx_pme_t *pme)
{
    const int n = pme->gpu->constants.nAtoms;
    assert(n > 0);
    const size_t forcesSize = DIM * n * sizeof(real);
    pme->gpu->forces = (real *)PMEMemoryFetch(pme, PME_ID_FORCES, forcesSize, ML_DEVICE);
}

// copy the PME resulting forces from the GPU to the CPU buffer
void pme_gpu_copy_forces(gmx_pme_t *pme)
{
    // host-to-device copy of the forces to be reduced with the gather results
    // we have to be sure that the atc forces (listed and such) are already calculated

    const int n = pme->gpu->constants.nAtoms;
    assert(n > 0);
    const int forcesSize = DIM * n * sizeof(real);
    real *forces = (real *)PMEMemoryFetch(pme, PME_ID_FORCES, forcesSize, ML_HOST);
    memcpy(forces, pme->atc[0].f, forcesSize);
    cu_copy_H2D_async(pme->gpu->forces, forces, forcesSize, pme->gpu->pmeStream);
}

// wait for the PME resulting forces on the CPU, and copy to the original CPU buffer (atc->f)
void pme_gpu_get_forces(gmx_pme_t *pme)
{
    cudaStream_t s = pme->gpu->pmeStream;
    cudaError_t stat = cudaStreamWaitEvent(s, pme->gpu->syncForcesD2H, 0);
    CU_RET_ERR(stat, "error while waiting for PME forces");

    const int n = pme->gpu->constants.nAtoms;
    assert(n > 0);
    const size_t forcesSize = DIM * n * sizeof(real);
    real *forces = (real *)PMEMemoryFetch(pme, PME_ID_FORCES, forcesSize, ML_HOST);
    memcpy(pme->atc[0].f, forces, forcesSize);
    // did not succeed in using cudaHostRegister instead of memcpy

    //pme_gpu_sloppy_force_reduction(pme, forces);
    // done on the GPU instead by passing bOverwriteForces = FALSE to the gather functions
}

// unroll the dynamic index accesses to the constant grid sizes to avoid local memory operations
__device__ __forceinline__ real read_grid_size(const pme_gpu_const_parameters constants, const int dimIndex)
{
    switch (dimIndex)
    {
        case XX: return constants.localGridSizeFP.x;
        case YY: return constants.localGridSizeFP.y;
        case ZZ: return constants.localGridSizeFP.z;
    }
    // we really shouldn't be here
    assert(false);
    return 0.0f;
}

// perform the gathering stage of the PME
template <
        const int order,                    // the PME interpolation order - assumed to be 4
        const int particlesPerBlock,        // the number of particles in a block - assumed to be (2 * number of warps) for order of 4
        const gmx_bool bOverwriteForces     // true: forces are written into the output, false: forces are added to the output
        >
__launch_bounds__(4 * warp_size, 16)
__global__ void pme_gather_kernel(const real * __restrict__ gridGlobal,          // global grid after the solving, overlap of (order - 1) included
                                  const pme_gpu_const_parameters constants,
                                  const real * __restrict__ thetaGlobal,
                                  const real * __restrict__ dthetaGlobal,
                                  real * __restrict__ forcesGlobal,
                                  const real * __restrict__ coefficientGlobal,
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
    __shared__ float2 splineParams[thetaSize]; // theta/dtheta pairs

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
        splineParams[threadLocalId].x = thetaGlobal[blockIdx.x * thetaSize + threadLocalId];
        splineParams[threadLocalId].y = dthetaGlobal[blockIdx.x * thetaSize + threadLocalId];
    }

    //locality?
    __syncthreads();

    real fx = 0.0f;
    real fy = 0.0f;
    real fz = 0.0f;

    if (globalIndex < constants.nAtoms)
    {
        const int pny = constants.localGridSizePadded.y;
        const int pnz = constants.localGridSizePadded.z;

        const int particleWarpIndex = localIndex % PARTICLES_PER_WARP;
        const int warpIndex = localIndex / PARTICLES_PER_WARP; // should be just a real warp index!

        const int thetaOffsetBase = PME_SPLINE_THETA_STRIDE * order * warpIndex * DIM * PARTICLES_PER_WARP + particleWarpIndex;
        const int orderStride = PME_SPLINE_THETA_STRIDE * DIM * PARTICLES_PER_WARP;  // PME_SPLINE_ORDER_STRIDE
        const int dimStride = PME_SPLINE_THETA_STRIDE * PARTICLES_PER_WARP; // ?

        const int thetaOffsetY = thetaOffsetBase + ithy * orderStride + YY * dimStride;
        const float2 tdy = splineParams[thetaOffsetY];
        const int thetaOffsetZ = thetaOffsetBase + ithz * orderStride + ZZ * dimStride;
        const float2 tdz = splineParams[thetaOffsetZ];
        const int indexBaseYZ = ((idx[localIndex * DIM + XX] + 0) * pny + (idx[localIndex * DIM + YY] + ithy)) * pnz + (idx[localIndex * DIM + ZZ] + ithz);
#pragma unroll
        for (int ithx = 0; (ithx < order); ithx++)
        {
            const real gridValue = gridGlobal[indexBaseYZ + ithx * pny * pnz];
            const int thetaOffsetX = thetaOffsetBase + ithx * orderStride + XX * dimStride;
            const float2 tdx = splineParams[thetaOffsetX];
            const real fxy1 = tdz.x * gridValue;
            const real fz1  = tdz.y * gridValue;
            fx += tdx.y * tdy.x * fxy1;
            fy += tdx.x * tdy.y * fxy1;
            fz += tdx.x * tdy.x * fz1;
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
        {
            const real n = read_grid_size(constants, splineIndex);
            *((real *)(&fSumArray[localIndex]) + splineIndex) = fx * n;
        }
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
            const real n = read_grid_size(constants, splineIndex);
            float f = 0.0f;
            for (int j = localIndex * particleDataSize; j < (localIndex + 1) * particleDataSize; j++)
            {
                f += fSharedArray[blockSize * splineIndex + j];
            }
            *((real *)(&fSumArray[localIndex]) + splineIndex) = f * n;
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
            contrib = constants.recipbox[XX].x * fSum.x /*+ constants.recipbox[YY].x * fSum.y + constants.recipbox[ZZ].x * fSum.z*/;
            break;

            case YY:
            contrib = constants.recipbox[XX].y * fSum.x + constants.recipbox[YY].y * fSum.y /* + constants.recipbox[ZZ].y * fSum.z*/;
            break;

            case ZZ:
            contrib = constants.recipbox[XX].z * fSum.x + constants.recipbox[YY].z * fSum.y + constants.recipbox[ZZ].z * fSum.z;
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
    (const pme_gpu_const_parameters constants,
        const struct pme_gpu_overlap_t OVERLAP,
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

    const int nx = constants.localGridSize.x;
    const int ny = constants.localGridSize.y;
    const int nz = constants.localGridSize.z;
    const int pny = constants.localGridSizePadded.y;
    const int pnz = constants.localGridSizePadded.z;


    // should use ldg.128

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
        const int ixy = cellIndex / zoneSizeYZ.y;
        // expensive integer divisions everywhere => should rewrite wrap/unwrap kernels
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
                   real scale,
                   const gmx_bool bOverwriteForces)
{
    int nAtoms = pme->gpu->constants.nAtoms; //spline->n;
    if (!nAtoms)
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

    const int nx = pme->gpu->constants.localGridSize.x;
    const int ny = pme->gpu->constants.localGridSize.y;
    const int nz = pme->gpu->constants.localGridSize.z;
    const int gridSize = pme->gpu->constants.localGridSizePadded.x *
            pme->gpu->constants.localGridSizePadded.y * pme->gpu->constants.localGridSizePadded.z * sizeof(real);
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

            pme_unwrap_kernel<4> <<<nBlocks, blockSize, 0, s>>>(pme->gpu->constants,
                                                                pme->gpu->overlap,
                                                                pme->gpu->grid);

            CU_LAUNCH_ERR("pme_unwrap_kernel");

            pme_gpu_timing_stop(pme, ewcsPME_UNWRAP);

        }
        else
            gmx_fatal(FARGS, "gather: orders other than 4 untested!");
    }

    int forcesSize = DIM * nAtoms * sizeof(real);
    int size_indices = nAtoms * sizeof(int);
    int size_splines = order * nAtoms * sizeof(int);

    real *atc_f_h = (real *)PMEMemoryFetch(pme, PME_ID_FORCES, forcesSize, ML_HOST);

    for (int i = 0; i < nAtoms; i++)
    {
        // coefficients
        atc_coefficient[i] *= scale;
    }

    // thetas
    real *theta_d = (real *)PMEMemoryFetch(pme, PME_ID_THETA, DIM * size_splines, ML_DEVICE);
    real *dtheta_d = (real *)PMEMemoryFetch(pme, PME_ID_DTHETA, DIM * size_splines, ML_DEVICE);

    // indices
    int *idx_d = (int *)PMEMemoryFetch(pme, PME_ID_IDXPTR, DIM * size_indices, ML_DEVICE);



    const int blockSize = 4 * warp_size;
    const int particlesPerBlock = blockSize / order / order;
    dim3 nBlocks((nAtoms + blockSize - 1) / blockSize * order * order);
    dim3 dimBlock(order, order, particlesPerBlock);

    pme_gpu_timing_start(pme, ewcsPME_GATHER);

    if (order == 4)
        if (bOverwriteForces)
            pme_gather_kernel<4, blockSize / 4 / 4, TRUE> <<<nBlocks, dimBlock, 0, s>>>
              (pme->gpu->grid,
               pme->gpu->constants,
               theta_d, dtheta_d,
               pme->gpu->forces, pme->gpu->coefficients,
               idx_d);
        else
            pme_gather_kernel<4, blockSize / 4 / 4, FALSE> <<<nBlocks, dimBlock, 0, s>>>
              (pme->gpu->grid,
               pme->gpu->constants,
               theta_d, dtheta_d,
               pme->gpu->forces, pme->gpu->coefficients,
               idx_d);
    else
        gmx_fatal(FARGS, "gather: orders other than 4 untested!");
    CU_LAUNCH_ERR("pme_gather_kernel");

    pme_gpu_timing_stop(pme, ewcsPME_GATHER);

    cu_copy_D2H_async(atc_f_h, pme->gpu->forces, forcesSize, s);
    cudaError_t stat = cudaEventRecord(pme->gpu->syncForcesD2H, s);
    CU_RET_ERR(stat, "PME gather forces sync fail");
}

