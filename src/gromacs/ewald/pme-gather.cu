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
    const int tag = 0;
    const int n = pme->atc[0].n; //?
    assert(n > 0);
    const int forcesSize = DIM * n * sizeof(real);
    pme->gpu->forces = (real *)PMEMemoryFetch(PME_ID_FORCES, tag, forcesSize, ML_DEVICE);
}

void pme_gpu_get_forces(gmx_pme_t *pme)
{
    cudaStream_t s = pme->gpu->pmeStream;
    cudaError_t stat = cudaStreamWaitEvent(s, pme->gpu->syncForcesH2D, 0);
    CU_RET_ERR(stat, "error while waiting for PME forces");

    const int tag = 0;
    const int n = pme->atc[0].n;
    const int forcesSize = DIM * n * sizeof(real);
    real *forces = (real *)PMEMemoryFetch(PME_ID_FORCES, tag, forcesSize, ML_HOST);
    memcpy(pme->atc[0].f, forces, forcesSize);

    /*
    if (PME_SKIP_ZEROES)
    {
        const int thread = 0;
        const int size_forces = DIM * n * sizeof(real);
        const int size_indices = n * sizeof(int);
        real *atc_f_h = (real *)PMEMemoryFetch(PME_ID_FORCES, thread, size_forces, ML_HOST);
        int *atc_i_compacted_h = (int *)PMEMemoryFetch(PME_ID_NONZERO_INDICES, thread, size_indices, ML_HOST);
        for (int iCompacted = 0; iCompacted < n; iCompacted++)  // iterating over compacted particles
        {
            int i = atc_i_compacted_h[iCompacted]; //index of uncompacted particle
            forces[i][XX] = atc_f_h[iCompacted * DIM + XX];
            forces[i][YY] = atc_f_h[iCompacted * DIM + YY];
            forces[i][ZZ] = atc_f_h[iCompacted * DIM + ZZ];
        }
    }
    */
}


//yupinov - texture memory?
template <
        const int order,
        const int particlesPerBlock,
        const gmx_bool bOverwriteForces
        >
__launch_bounds__(4 * warp_size, 16)
__global__ void pme_gather_kernel
(const real * __restrict__ gridGlobal, const int n,
 const float3 nXYZ, const int pnx, const int pny, const int pnz,
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

    const int thetaStride = particlesPerBlock * DIM; // a global size dependency with spread!
    const int thetaSize = thetaStride * order;
    const int idxSize = thetaStride;
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
        const int thetaOffsetY = localIndex * DIM + ithy * thetaStride + YY;
        const int thetaOffsetZ = localIndex * DIM + ithz * thetaStride + ZZ;
        const real ty = theta[thetaOffsetY];
        const real tz = theta[thetaOffsetZ];
        const real dy = dtheta[thetaOffsetY];
        const real dz = dtheta[thetaOffsetZ];
        //yupinov need to reorder theta when transferring thetas to and from CPU!
        for (int ithx = 0; (ithx < order); ithx++)
        {
            const int index_x = (idx[localIndex * DIM + XX] + ithx) * pny * pnz;
            const int index_xy = index_x + (idx[localIndex * DIM + YY] + ithy) * pnz;
            const real gridValue = gridGlobal[index_xy + (idx[localIndex * DIM + ZZ] + ithz)];
            const int thetaOffsetX = localIndex * DIM + ithx * thetaStride + XX;
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
    __syncthreads(); // breaking globalIndex condition?

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

        // a single operation for all 3 components!
        if (splineIndex < 3)
            *((real *)(&fSumArray[localIndex]) + splineIndex) = fx * ((real *)&nXYZ)[splineIndex];
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
            *((real *)(&fSumArray[localIndex]) + splineIndex) = f * ((real *)&nXYZ)[splineIndex];
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
        // by columns!
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
            forcesGlobal[blockIdx.x * particlesPerBlock * DIM + threadLocalId] = contrib;
        else
            forcesGlobal[blockIdx.x * particlesPerBlock * DIM + threadLocalId] += contrib;
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
                   real scale)
{
    //yupinov bClearf!
    int n = spline->n;
    if (!n)
        return;

    const int thread = 0;

    const gmx_bool bOverwriteForces = true;
    // false: we use some other GPU forces buffer for the final reduction, so we want to add to that
    // in that case, maybe we want to replace + with atomicAdd at the end of kernel?
    // true: we have our own buffer, so just write directly into that


    const int *spline_ind = spline->ind;
    const splinevec *spline_theta = &spline->theta;
    const splinevec *spline_dtheta = &spline->dtheta;

    cudaStream_t s = pme->gpu->pmeStream;

    //pme_atomcomm_t atc = pme->atc[0];
    real *atc_coefficient = atc->coefficient;
    ivec *atc_idx = atc->idx;


    const int order = pme->pme_order;
    /*
    gmx_parallel_3dfft_real_limits_wrapper(pme, grid_index, local_ndata, local_offset, local_size);
    const int pnx = local_size[XX];
    const int pny = local_size[YY];
    const int pnz = local_size[ZZ];
    const int nx = local_ndata[XX];
    const int ny = local_ndata[YY];
    const int nz = local_ndata[ZZ];
    */

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
    if (!pme->gpu->keepGPUDataBetweenC2RAndGather)
        cu_copy_H2D_async(pme->gpu->grid, grid, gridSize, s);

    if (pme->bGPUSingle)
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
    int size_coefficients = n * sizeof(real);

    real *atc_f_h = (real *)PMEMemoryFetch(PME_ID_FORCES, thread, forcesSize, ML_HOST);
    ivec *idx_h = NULL;

    real *coefficients_h = NULL;

    real *theta_x_h = NULL, *theta_y_h = NULL, *theta_z_h = NULL;
    real *dtheta_x_h = NULL, *dtheta_y_h = NULL, *dtheta_z_h = NULL;

    /*
    int *i0_h = NULL, *j0_h = NULL, *k0_h = NULL;
    i0_h = (int *)PMEFetch(PME_ID_I0, thread, size_indices, ML_HOST);
    j0_h = (int *)PMEFetch(PME_ID_J0, thread, size_indices, ML_HOST);
    k0_h = (int *)PMEFetch(PME_ID_K0, thread, size_indices, ML_HOST);
    */

    int *atc_i_compacted_h = NULL;

    // compact data (might be broken)
    if (PME_SKIP_ZEROES)
    {
        atc_i_compacted_h = (int *)PMEMemoryFetch(PME_ID_NONZERO_INDICES, thread, size_indices, ML_HOST);

        // thetas
        theta_x_h = (real *)PMEMemoryFetch(PME_ID_THX, thread, size_splines, ML_HOST);
        theta_y_h = (real *)PMEMemoryFetch(PME_ID_THY, thread, size_splines, ML_HOST);
        theta_z_h = (real *)PMEMemoryFetch(PME_ID_THZ, thread, size_splines, ML_HOST);
        dtheta_x_h = (real *)PMEMemoryFetch(PME_ID_DTHX, thread, size_splines, ML_HOST);
        dtheta_y_h = (real *)PMEMemoryFetch(PME_ID_DTHY, thread, size_splines, ML_HOST);
        dtheta_z_h = (real *)PMEMemoryFetch(PME_ID_DTHZ, thread, size_splines, ML_HOST);

        // indices
        idx_h = (ivec *)(int *)PMEMemoryFetch(PME_ID_IDXPTR, thread, DIM * size_indices, ML_HOST);

        // coefficients
        coefficients_h = (real *)PMEMemoryFetch(PME_ID_COEFFICIENT, thread, size_coefficients, ML_HOST);

        int iCompacted = 0;
        for (int ii = 0; ii < n; ii++)
        {
            int iOriginal = spline_ind[ii]; //should be just 1 : 1

            // coefficients
            real coefficient_i = scale * atc_coefficient[iOriginal]; //yupinov mutiply coefficients on device!

            if (coefficient_i != 0.0f)
            {
                coefficients_h[iCompacted] = coefficient_i;

                //indices
                /*
                int *idxptr = atc_idx[iOriginal];
                i0_h[iCompacted] = idxptr[XX];
                j0_h[iCompacted] = idxptr[YY];
                k0_h[iCompacted] = idxptr[ZZ];
                */
                memcpy(idx_h + iCompacted, atc_idx + iOriginal, sizeof(ivec));

                // thetas
                int iiorder = ii * order;
                int ooorder = iCompacted * order;
                for (int o = 0; o < order; ++o)
                {
                    theta_x_h[ooorder + o] = (*spline_theta)[XX][iiorder + o];
                    theta_y_h[ooorder + o] = (*spline_theta)[YY][iiorder + o];
                    theta_z_h[ooorder + o] = (*spline_theta)[ZZ][iiorder + o];
                    dtheta_x_h[ooorder + o] = (*spline_dtheta)[XX][iiorder + o];
                    dtheta_y_h[ooorder + o] = (*spline_dtheta)[YY][iiorder + o];
                    dtheta_z_h[ooorder + o] = (*spline_dtheta)[ZZ][iiorder + o];
                }

                // indices of uncompacted particles stored in a compacted array
                atc_i_compacted_h[iCompacted] = iOriginal;

                iCompacted++;
            }
        }
        // adjust sizes for device allocation
        n = iCompacted;
        size_coefficients = n * sizeof(real);
        size_splines = order * n * sizeof(int);
        size_indices = n * sizeof(int);
        forcesSize = DIM * n * sizeof(real);
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            // indices
            /*
            i0_h[i] = atc_idx[i][XX];
            j0_h[i] = atc_idx[i][YY];
            k0_h[i] = atc_idx[i][ZZ];
            */

            // coefficients
            atc_coefficient[i] *= scale;
        }

        // indices
        idx_h = atc_idx;
        // coefficients
        coefficients_h = atc_coefficient;
        // thetas
        theta_x_h = (*spline_theta)[XX];
        theta_y_h = (*spline_theta)[YY];
        theta_z_h = (*spline_theta)[ZZ];
        dtheta_x_h = (*spline_dtheta)[XX];
        dtheta_y_h = (*spline_dtheta)[YY];
        dtheta_z_h = (*spline_dtheta)[ZZ];
    }

    // thetas
    /*
    real *theta_x_d = (real *)PMEFetchAndCopy(PME_ID_THX, thread, theta_x_h, size_splines, s);
    real *theta_y_d = (real *)PMEFetchAndCopy(PME_ID_THY, thread, theta_y_h, size_splines, s);
    real *theta_z_d = (real *)PMEFetchAndCopy(PME_ID_THZ, thread, theta_z_h, size_splines, s);
    real *dtheta_x_d = (real *)PMEFetchAndCopy(PME_ID_DTHX, thread, dtheta_x_h, size_splines, s);
    real *dtheta_y_d = (real *)PMEFetchAndCopy(PME_ID_DTHY, thread, dtheta_y_h, size_splines, s);
    real *dtheta_z_d = (real *)PMEFetchAndCopy(PME_ID_DTHZ, thread, dtheta_z_h, size_splines, s);
    */
    real *theta_d = (real *)PMEMemoryFetch(PME_ID_THETA, thread, DIM * size_splines, ML_DEVICE);
    real *theta_x_d = theta_d + 0 * order * n;
    real *theta_y_d = theta_d + 1 * order * n;
    real *theta_z_d = theta_d + 2 * order * n;

    real *dtheta_d = (real *)PMEMemoryFetch(PME_ID_DTHETA, thread, DIM * size_splines, ML_DEVICE);
    real *dtheta_x_d = dtheta_d + 0 * order * n;
    real *dtheta_y_d = dtheta_d + 1 * order * n;
    real *dtheta_z_d = dtheta_d + 2 * order * n;

    // coefficients
    //real *coefficients_d = (real *)PMEMemoryFetch(PME_ID_COEFFICIENT, thread, size_coefficients, ML_DEVICE);
    //yupinov

    // indices
    int *idx_d = (int *)PMEMemoryFetch(PME_ID_IDXPTR, thread, DIM * size_indices, ML_DEVICE);

    if (!pme->gpu->keepGPUDataBetweenC2RAndGather) // compare with spread and compacting
    {
        cu_copy_H2D_async(theta_x_d, theta_x_h, size_splines, s);
        cu_copy_H2D_async(theta_y_d, theta_y_h, size_splines, s);
        cu_copy_H2D_async(theta_z_d, theta_z_h, size_splines, s);

        cu_copy_H2D_async(dtheta_x_d, dtheta_x_h, size_splines, s);
        cu_copy_H2D_async(dtheta_y_d, dtheta_y_h, size_splines, s);
        cu_copy_H2D_async(dtheta_z_d, dtheta_z_h, size_splines, s);

        //yupinov cu_copy_H2D_async(pme->gpu->coefficients, coefficients_h, size_coefficients, s);

        cu_copy_H2D_async(idx_d, idx_h, DIM * size_indices, s);
    }

    const float3 nXYZ = {(real)nx, (real)ny, (real)nz};


    const int blockSize = 4 * warp_size;
    const int particlesPerBlock = blockSize / order / order;
    dim3 nBlocks((n + blockSize - 1) / blockSize * order * order); //yupinov what does this mean?
    dim3 dimBlock(order, order, particlesPerBlock);

    pme_gpu_timing_start(pme, ewcsPME_GATHER);

    if (order == 4) //yupinov
        if (bOverwriteForces)
            pme_gather_kernel<4, blockSize / 4 / 4, TRUE> <<<nBlocks, dimBlock, 0, s>>>
              (pme->gpu->grid,
               n,
               nXYZ, pnx, pny, pnz,
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
               nXYZ, pnx, pny, pnz,
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
    cudaError_t stat = cudaEventRecord(pme->gpu->syncForcesH2D, s);
    CU_RET_ERR(stat, "PME gather forces sync fail");
}

