/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

/*! \internal \file
 *  \brief Implements PME GPU charge gathering in CUDA.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"
#include <assert.h>
#include "gromacs/utility/gmxassert.h"
#include "pme.cuh"

/*! \brief
 *
 * Copies the forces from the CPU buffer (pme->gpu->forcesHost) to the GPU
 * (to reduce them with the PME GPU gathered forces).
 * To be called after the bonded calculations.
 * FIXME: either this functon goes to the pme.cu, or the other functions go out of the pme.cu...
 */
void pme_gpu_copy_input_forces(const gmx_pme_t *pme)
{
    GMX_ASSERT(pme->gpu->forcesHost, "NULL host forces pointer in PME GPU");
    const size_t forcesSize = DIM * pme->gpu->kernelParams.atoms.nAtoms * sizeof(float);
    assert(forcesSize > 0);
    cu_copy_H2D_async(pme->gpu->kernelParams.atoms.forces, pme->gpu->forcesHost, forcesSize, pme->gpu->pmeStream);
}

void pme_gpu_sync_output_forces(const gmx_pme_t *pme)
{
    cudaStream_t s    = pme->gpu->pmeStream;
    cudaError_t  stat = cudaStreamWaitEvent(s, pme->gpu->syncForcesD2H, 0);
    CU_RET_ERR(stat, "Error while waiting for the PME GPU forces");

    for (int i = 0; i < DIM * pme->gpu->kernelParams.atoms.nAtoms; i++)
    {
        GMX_ASSERT(!isnan(pme->gpu->forcesHost[i]), "PME GPU - wrong forces produced.");
    }
}

/*! \brief
 *
 * An inline CUDA function: unroll the dynamic index accesses to the constant grid sizes to avoid local memory operations.
 */
__device__ __forceinline__ float read_grid_size(const float3 localGridSizeFP,
                                                const int    dimIndex)
{
    switch (dimIndex)
    {
        case XX: return localGridSizeFP.x;
        case YY: return localGridSizeFP.y;
        case ZZ: return localGridSizeFP.z;
    }
    assert(false);
    return 0.0f;
}

/*! \brief Reduce the order^2 contributions.
 *
 *  \param fSumArray[in]    shared memory array with the partial contributions
 *  \param localIndex[in]   local index
 *  \param splineIndex[in]  spline index
 *  \param lineIndex[in]    line index
 *  \param localGridSizeFP[in]   local grid zsize constant
 *  \param fx[out]          force x component
 *  \param fy[out]          force y component
 *  \param fz[out]          force z component
 *
 */
template <
    const int order,
    const int particleDataSize,
    const int blockSize
    >
__device__ __forceinline__ void reduce_particle_forces(float3         fSumArray[],
                                                       const int      localIndex,
                                                       const int      splineIndex,
                                                       const int      lineIndex,
                                                       const float3   localGridSizeFP,
                                                       float         &fx,
                                                       float         &fy,
                                                       float         &fz)
{
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
            const float n = read_grid_size(localGridSizeFP, splineIndex);
            *((float *)(&fSumArray[localIndex]) + splineIndex) = fx * n;
        }
    }
    else
#endif
    {
        // TODO (psz): improve the generic reduction
        // lazy 3-thread reduction in shared memory inspired by reduce_force_j_generic
        __shared__ float fSharedArray[DIM * blockSize];
        fSharedArray[XX * blockSize + lineIndex] = fx;
        fSharedArray[YY * blockSize + lineIndex] = fy;
        fSharedArray[ZZ * blockSize + lineIndex] = fz;

        if (splineIndex < 3)
        {
            const float n = read_grid_size(localGridSizeFP, splineIndex);
            float       f = 0.0f;
            for (int j = localIndex * particleDataSize; j < (localIndex + 1) * particleDataSize; j++)
            {
                f += fSharedArray[blockSize * splineIndex + j];
            }
            *((float *)(&fSumArray[localIndex]) + splineIndex) = f * n;
        }
    }
}

/*! \brief
 *
 * A CUDA kernel: gathers the forces from the grid in the last PME GPU stage.
 *
 * Template parameters:
 * \tparam[in] order                The PME order (must be 4).
 * \tparam[in] particlesPerBlock    The number of particles processed by a single block;
 *                                  currently this is (warp_size / order^2) * (number of warps in a block) = (32 / 16) * 4 = 8.
 * \tparam[in] bOverwriteForces     TRUE: the forces are written to the output buffer;
 *                                  FALSE: the forces are added non-atomically to the output buffer (e.g. to the bonded forces).
 *
 * Normal parameters:
 * \param[in] kernelParams          All the PME GPU data.
 * \param[out] forcesGlobal         The rvec forces for the output, sorted by particles.
 */
template <
    const int order,
    const int particlesPerBlock,
    const gmx_bool bOverwriteForces
    >
__launch_bounds__(4 * warp_size, 16)
__global__ void pme_gather_kernel(const pme_gpu_kernel_params    kernelParams)
{
    /* Global memory pointers */
    const float * __restrict__  coefficientGlobal     = kernelParams.atoms.coefficients;
    const float * __restrict__  gridGlobal            = kernelParams.grid.realGrid;
    const float * __restrict__  thetaGlobal           = kernelParams.atoms.theta;
    const float * __restrict__  dthetaGlobal          = kernelParams.atoms.dtheta;
    const int * __restrict__    gridlineIndicesGlobal = kernelParams.atoms.gridlineIndices;
    float * __restrict__        forcesGlobal          = kernelParams.atoms.forces;


    /* These are the atom indices - for the shared and global memory */
    const int localIndexGather  = threadIdx.z;
    const int globalIndexGather = blockIdx.x * blockDim.z + threadIdx.z;

    const int particleDataSize = order * order; /* Number of data components and threads for a single particle */
    const int blockSize        = particlesPerBlock * particleDataSize;
    // should the array size aligned by warp size for shuffle?

    const int                 splineParamsSize             = PME_SPREADGATHER_BLOCK_DATA_SIZE * order;
    const int                 gridlineIndicesSize          = PME_SPREADGATHER_BLOCK_DATA_SIZE;
    __shared__ int            gridlineIndices[gridlineIndicesSize];
    __shared__ float2         splineParams[splineParamsSize]; /* Theta/dtheta pairs */

    /* Spline Y/Z coordinates */
    const int ithy = threadIdx.y;
    const int ithz = threadIdx.x;
    /* These are the spline contribution indices in shared memory */
    const int splineIndex = threadIdx.y * blockDim.x + threadIdx.x;                  /* Relative to the current particle , 0..15 for order 4 */
    const int lineIndex   = (threadIdx.z * (blockDim.x * blockDim.y)) + splineIndex; /* And to all the block's particles */

    int       threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;

    /* Staging the atom gridline indices, DIM * particlesPerBlock = 24 threads */
    const int localGridlineIndicesIndex  = threadLocalId;
    const int globalGridlineIndicesIndex = blockIdx.x * gridlineIndicesSize + localGridlineIndicesIndex;
    const int globalCheckIndices         = pme_gpu_check_atom_data_index(globalGridlineIndicesIndex, kernelParams.atoms.nAtoms * DIM);
    if ((localGridlineIndicesIndex < gridlineIndicesSize) & globalCheckIndices)
    {
        gridlineIndices[localGridlineIndicesIndex] = gridlineIndicesGlobal[globalGridlineIndicesIndex];
    }
    /* Staging the spline parameters, DIM * order * particlesPerBlock = 96 threads */
    const int localSplineParamsIndex  = threadLocalId;
    const int globalSplineParamsIndex = blockIdx.x * splineParamsSize + localSplineParamsIndex;
    const int globalCheckSplineParams = pme_gpu_check_atom_data_index(globalSplineParamsIndex, kernelParams.atoms.nAtoms * DIM * order);
    if ((localSplineParamsIndex < splineParamsSize) && globalCheckSplineParams)
    {
        splineParams[localSplineParamsIndex].x = thetaGlobal[globalSplineParamsIndex];
        splineParams[localSplineParamsIndex].y = dthetaGlobal[globalSplineParamsIndex];
    }
    __syncthreads();

    float           fx = 0.0f;
    float           fy = 0.0f;
    float           fz = 0.0f;

    const int       globalCheck = pme_gpu_check_atom_data_index(globalIndexGather, kernelParams.atoms.nAtoms);
    const int       chargeCheck = pme_gpu_check_atom_charge(coefficientGlobal[globalIndexGather]);

    //yupinov stage coefficient into a shared/local mem?
    if (chargeCheck & globalCheck)
    {
        const int    pny = kernelParams.grid.localGridSizePadded.y;
        const int    pnz = kernelParams.grid.localGridSizePadded.z;

        const int    particleWarpIndex = localIndexGather % PME_SPREADGATHER_PARTICLES_PER_WARP;
        const int    warpIndex         = localIndexGather / PME_SPREADGATHER_PARTICLES_PER_WARP;

        const int    thetaOffsetBase = PME_SPLINE_THETA_STRIDE * order * warpIndex * DIM * PME_SPREADGATHER_PARTICLES_PER_WARP + particleWarpIndex;
        const int    orderStride     = PME_SPLINE_THETA_STRIDE * DIM * PME_SPREADGATHER_PARTICLES_PER_WARP; // PME_SPLINE_ORDER_STRIDE
        const int    dimStride       = PME_SPLINE_THETA_STRIDE * PME_SPREADGATHER_PARTICLES_PER_WARP;

        const int    thetaOffsetY = thetaOffsetBase + ithy * orderStride + YY * dimStride;
        const float2 tdy          = splineParams[thetaOffsetY];
        const int    thetaOffsetZ = thetaOffsetBase + ithz * orderStride + ZZ * dimStride;
        const float2 tdz          = splineParams[thetaOffsetZ];
        const int    indexBaseYZ  = ((gridlineIndices[localIndexGather * DIM + XX] + 0) * pny + (gridlineIndices[localIndexGather * DIM + YY] + ithy)) * pnz + (gridlineIndices[localIndexGather * DIM + ZZ] + ithz);
#pragma unroll
        for (int ithx = 0; (ithx < order); ithx++)
        {
            const float   gridValue    = gridGlobal[indexBaseYZ + ithx * pny * pnz];
            assert(!isnan(gridValue));
            const int     thetaOffsetX = thetaOffsetBase + ithx * orderStride + XX * dimStride;
            const float2  tdx          = splineParams[thetaOffsetX];
            const float   fxy1         = tdz.x * gridValue;
            const float   fz1          = tdz.y * gridValue;
            fx += tdx.y * tdy.x * fxy1;
            fy += tdx.x * tdy.y * fxy1;
            fz += tdx.x * tdy.x * fz1;
        }
    }
    __syncthreads();

    // now (particlesPerBlock) particles have to reduce (order^2) contributions each
    __shared__ float3 fSumArray[particlesPerBlock];
    reduce_particle_forces<order, particleDataSize, blockSize>(fSumArray,
                                                               localIndexGather, splineIndex, lineIndex,
                                                               kernelParams.grid.localGridSizeFP,
                                                               fx, fy, fz);
    __syncthreads();

    /* Calculating the final forces with no component branching, particlesPerBlock = 8 threads */
    const int localCalcIndex  = threadLocalId;
    const int globalCalcIndex = blockIdx.x * particlesPerBlock + localCalcIndex;
    const int globalCalcCheck = pme_gpu_check_atom_data_index(globalCalcIndex, kernelParams.atoms.nAtoms);
    if ((localCalcIndex < particlesPerBlock) && globalCalcCheck)
    {
        const float3  fSum               = fSumArray[localCalcIndex];
        const float   negCoefficient     = -coefficientGlobal[globalCalcIndex];
        float3        result;
        result.x                  = negCoefficient * kernelParams.step.recipBox[XX].x * fSum.x;
        result.y                  = negCoefficient * (kernelParams.step.recipBox[XX].y * fSum.x + kernelParams.step.recipBox[YY].y * fSum.y);
        result.z                  = negCoefficient * (kernelParams.step.recipBox[XX].z * fSum.x + kernelParams.step.recipBox[YY].z * fSum.y + kernelParams.step.recipBox[ZZ].z * fSum.z);
        fSumArray[localCalcIndex] = result;
    }
    /* Writing out the final forces, DIM * particlesPerBlock = 24 threads */
    const int localOutputIndex  = threadLocalId;
    const int globalOutputIndex = blockIdx.x * PME_SPREADGATHER_BLOCK_DATA_SIZE + localOutputIndex;
    const int globalOutputCheck = pme_gpu_check_atom_data_index(globalOutputIndex, kernelParams.atoms.nAtoms * DIM);
    if ((localOutputIndex < particlesPerBlock * DIM) && globalOutputCheck)
    {
        float outputForceComponent = ((float *)fSumArray)[localOutputIndex];
        if (bOverwriteForces)
        {
            forcesGlobal[globalOutputIndex] = outputForceComponent;
        }
        else
        {
            forcesGlobal[globalOutputIndex] += outputForceComponent;
        }
    }
}

// a quick dirty copy of pme_wrap_kernel
template <
    const int order
    >
__global__ void pme_unwrap_kernel(const pme_gpu_kernel_params kernelParams)
{
    /* Global memory pointer */
    float * __restrict__  gridGlobal = kernelParams.grid.realGrid;

    int                   blockId = blockIdx.x
        + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;

    const int nx  = kernelParams.grid.localGridSize.x;
    const int ny  = kernelParams.grid.localGridSize.y;
    const int nz  = kernelParams.grid.localGridSize.z;
    const int pny = kernelParams.grid.localGridSizePadded.y;
    const int pnz = kernelParams.grid.localGridSizePadded.z;


    // should use ldg.128

    if (threadId < kernelParams.grid.overlapCellCounts[PME_GPU_OVERLAP_ZONES_COUNT - 1])
    {
        int zoneIndex = -1;
        do
        {
            zoneIndex++;
        }
        while (threadId >= kernelParams.grid.overlapCellCounts[zoneIndex]);
        const int2 zoneSizeYZ = kernelParams.grid.overlapSizes[zoneIndex];
        // this is the overlapped cells's index relative to the current zone
        const int  cellIndex = (zoneIndex > 0) ? (threadId - kernelParams.grid.overlapCellCounts[zoneIndex - 1]) : threadId;

        // replace integer division/modular arithmetics - a big performance hit
        // try int_fastdiv?
        const int ixy = cellIndex / zoneSizeYZ.y;
        // expensive integer divisions everywhere => should rewrite wrap/unwrap kernels
        const int iz          = cellIndex - zoneSizeYZ.y * ixy;
        const int ix          = ixy / zoneSizeYZ.x;
        const int iy          = ixy - zoneSizeYZ.x * ix;
        const int sourceIndex = (ix * pny + iy) * pnz + iz;

        int       targetOffset = 0;

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
        gridGlobal[targetIndex] = gridGlobal[sourceIndex];
    }
}

void pme_gpu_gather(const gmx_pme_t *pme,
                    const gmx_bool   bOverwriteForces)
{
    /* Copying the input CPU forces for reduction */
    if (!bOverwriteForces)
    {
        pme_gpu_copy_input_forces(pme);
    }

    cudaStream_t s = pme->gpu->pmeStream;

    const int    order = pme->pme_order;

    if (!pme->gpu->bGPUFFT)
    {
        /* Copying the input CPU grid */
        const int       grid_index = 0;
        float          *grid       = pme->pmegrid[grid_index].grid.grid;
        const size_t    gridSize   = pme->gpu->kernelParams.grid.localGridSizePadded.x *
            pme->gpu->kernelParams.grid.localGridSizePadded.y * pme->gpu->kernelParams.grid.localGridSizePadded.z * sizeof(float);
        cu_copy_H2D_async(pme->gpu->kernelParams.grid.realGrid, grid, gridSize, s);
    }

    if (pme_gpu_performs_wrapping(pme))
    {
        /* The wrapping kernel */
        const int    blockSize = 4 * warp_size; //yupinov thsi is everywhere! and architecture-specific
        const int    overlap   = order - 1;

        const int    nx              = pme->gpu->kernelParams.grid.localGridSize.x;
        const int    ny              = pme->gpu->kernelParams.grid.localGridSize.y;
        const int    nz              = pme->gpu->kernelParams.grid.localGridSize.z;
        const int    overlappedCells = (nx + overlap) * (ny + overlap) * (nz + overlap) - nx * ny * nz;
        const int    nBlocks         = (overlappedCells + blockSize - 1) / blockSize;

        if (order == 4)
        {
            pme_gpu_start_timing(pme, gtPME_UNWRAP);
            pme_unwrap_kernel<4> <<< nBlocks, blockSize, 0, s>>> (pme->gpu->kernelParams);
            CU_LAUNCH_ERR("pme_unwrap_kernel");
            pme_gpu_stop_timing(pme, gtPME_UNWRAP);

        }
        else
        {
            gmx_fatal(FARGS, "PME GPU unwrapping: orders other than 4 not implemented!");
        }
    }

    /* The gathering kernel */
    const int blockSize         = 4 * warp_size;
    const int particlesPerBlock = blockSize / order / order;
    dim3 nBlocks(pme->gpu->nAtomsPadded / particlesPerBlock);
    dim3 dimBlock(order, order, particlesPerBlock);

    pme_gpu_start_timing(pme, gtPME_GATHER);
    if (order == 4)
    {
        if (bOverwriteForces)
        {
            pme_gather_kernel<4, blockSize / 4 / 4, TRUE> <<< nBlocks, dimBlock, 0, s>>> (pme->gpu->kernelParams);
        }
        else
        {
            pme_gather_kernel<4, blockSize / 4 / 4, FALSE> <<< nBlocks, dimBlock, 0, s>>> (pme->gpu->kernelParams);
        }
    }
    else
    {
        gmx_fatal(FARGS, "PME GPU gathering: orders other than 4 not implemented!");
    }
    CU_LAUNCH_ERR("pme_gather_kernel");
    pme_gpu_stop_timing(pme, gtPME_GATHER);

    /* Copying the output forces */
    const size_t forcesSize   = DIM * pme->gpu->kernelParams.atoms.nAtoms * sizeof(float);
    cu_copy_D2H_async(pme->gpu->forcesHost, pme->gpu->kernelParams.atoms.forces, forcesSize, s);
    cudaError_t  stat = cudaEventRecord(pme->gpu->syncForcesD2H, s);
    CU_RET_ERR(stat, "PME gather forces sync fail");
}
