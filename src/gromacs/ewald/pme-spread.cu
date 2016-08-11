/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013-2015, by the GROMACS development team, led by
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

#include "pme.h"
#include "pme-internal.h"

#include "gromacs/utility/fatalerror.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"

#include "pme-timings.cuh"
#include "pme-cuda.cuh"

#define PME_GPU_PARALLEL_SPLINE 1
// this define affects the spline calculation in the spreading kernel
// 0: a single thread handles a single dimension of a single particle (calculating and storing (order) spline values and derivatives)
// 1: (order) threads work on the same task, each one stores only a single theta and single dtheta into global arrays
// the only efficiency difference is less global store operations - and it matters!
// also, in a second case a data[order] was abusing local cache => replaced by shared array
// with PME_GPU_PARALLEL_SPLINE==1  the kernel is faster, but it doesn't guarantee that the whole program is faster!

//yupinov - describe theta layout properly somewhere!
/*
    here is a current memory layout for theta/dtheta spline parameter arrays
    this example has PME order 4 (the only order implemented and tested) and 2 particles per warp/data chunk
    ----------------------------------------------------------------------------
    particles 0, 1                                        | particles 2, 3 | ...
    ----------------------------------------------------------------------------
    order index 0           | index 1 | index 2 | index 3 | order 0 .....
    ----------------------------------------------------------------------------
    tx1 tx2 ty1 ty2 tz1 tz2 | ..........
    ----------------------------------------------------------------------------
    so each data chunk for a single warp is 24 floats - goes both for theta and dtheta
    24 = 2 particles per warp *  order 4 * 3 dimensions
    48 floats (1.5 warp size) per warp in total
*/

#define THREADS_PER_BLOCK   (4 * warp_size)
#define MIN_BLOCKS_PER_MP   (16)

//move all this into structure?
#if PME_USE_TEXTURES
#define USE_TEXOBJ 0 // should check device info dynamically
#if USE_TEXOBJ
cudaTextureObject_t nnTexture;
cudaTextureObject_t fshTexture;
#else
texture<int, 1, cudaReadModeElementType> nnTextureRef;
texture<float, 1, cudaReadModeElementType> fshTextureRef;
#endif
#endif

template <
        const int order,
        const int particlesPerBlock,
        const gmx_bool bCalcAlways
        >
__device__ __forceinline__ void calculate_splines(const int3 nnOffset,
#if PME_USE_TEXTURES
#if USE_TEXOBJ
                                        cudaTextureObject_t nnTexture,
                                        cudaTextureObject_t fshTexture,
#endif
#else
                                        const int * __restrict__ nn,
                                        const real * __restrict__ fsh,
#endif
                                        const float3 * __restrict__ coordinates,
                                        real * __restrict__ coefficient,
                                        real * __restrict__ thetaGlobal,
                                        real * __restrict__ theta,
                                        real * __restrict__ dthetaGlobal,
                                        int * __restrict__ idxGlobal,
                                        int * __restrict__ idx,
                                        const pme_gpu_const_parameters constants,
                                        const int globalIndexCalc,
                                        const int localIndexCalc,
                                        const int globalIndexBase,
                                        const int dimIndex,
                                        const int orderIndex)
{

    // fractional coordinates
    __shared__ real fractX[PME_SPREADGATHER_BLOCK_DATA_SIZE];

    const int sharedMemoryIndex = localIndexCalc * DIM + dimIndex;

    const int dataSize = PME_GPU_PARALLEL_SPLINE ? PME_SPREADGATHER_BLOCK_DATA_SIZE : 1;
    const int dataOffset = PME_GPU_PARALLEL_SPLINE ? sharedMemoryIndex : 0;
#if PME_GPU_PARALLEL_SPLINE
    __shared__
#endif
    real data[dataSize * order];

    const int localLimit = (dimIndex < DIM) && (orderIndex < (PME_GPU_PARALLEL_SPLINE ? order : 1));
    const int globalLimit = (globalIndexCalc < constants.nAtoms);

    // INTERPOLATION INDICES
    if (localLimit && globalLimit)
    {
        if (orderIndex == 0)
        {
            int constIndex, tInt;
            real n, t;
            const float3 x = coordinates[localIndexCalc];
            // accessing fields in nnOffset/nXYZ/recipbox/... with dimIndex offset
            // puts them into local memory (!) instead of accessing the constant memory directly
            // that's the reason for the switch
            switch (dimIndex)
            {
                case 0:
                constIndex = nnOffset.x;
                n = constants.gridSizeFP.x;
                t = x.x * constants.recipbox[dimIndex].x + x.y * constants.recipbox[dimIndex].y + x.z * constants.recipbox[dimIndex].z;
                break;

                case 1:
                constIndex = nnOffset.y;
                n = constants.gridSizeFP.y;
                t = /*x.x * constants.recipbox[dimIndex].x + */ x.y * constants.recipbox[dimIndex].y + x.z * constants.recipbox[dimIndex].z;
                break;

                case 2:
                constIndex = nnOffset.z;
                n = constants.gridSizeFP.z;
                t = /*x.x * constants.recipbox[dimIndex].x + x.y * constants.recipbox[dimIndex].y + */ x.z * constants.recipbox[dimIndex].z;
                break;
            }
            // parts of multiplication are commented because these components are actually 0
            // thus, excessive constant memory
            // should refactor if settling for this approach...

            // Fractional coordinates along box vectors, add 2.0 to make 100% sure we are positive for triclinic boxes
            t = (t + 2.0f) * n;
            tInt = (int)t;
            fractX[sharedMemoryIndex] = t - tInt;
            constIndex += tInt;

#if PME_USE_TEXTURES
#if USE_TEXOBJ
            fractX[sharedMemoryIndex] += tex1Dfetch<real>(fshTexture, constIndex);
            idx[sharedMemoryIndex] = tex1Dfetch<int>(nnTexture, constIndex);
#else
            fractX[sharedMemoryIndex] += tex1Dfetch(fshTextureRef, constIndex);
            idx[sharedMemoryIndex] = tex1Dfetch(nnTextureRef, constIndex);
#endif
#else
            fractX[sharedMemoryIndex] += fsh[constIndex];
            idx[sharedMemoryIndex] = nn[constIndex];
#endif
            // staging for both parts
            idxGlobal[globalIndexBase * DIM + sharedMemoryIndex] = idx[sharedMemoryIndex];
        }

        // MAKE BSPLINES

        if (bCalcAlways || (coefficient[localIndexCalc] != 0.0f))
        {
            real div;
            int k;

            const real dr = fractX[sharedMemoryIndex];

            /* dr is relative offset from lower cell limit */
            data[(order - 1) * dataSize + dataOffset] = 0.0f;
            data[1 * dataSize + dataOffset]         = dr;
            data[0 * dataSize + dataOffset]         = 1.0f - dr;

#pragma unroll
            for (int k = 3; k < order; k++)
            {
                div         = 1.0f / (k - 1.0f);
                data[(k - 1) * dataSize + dataOffset] = div * dr * data[(k - 2) * dataSize + dataOffset];
#pragma unroll
                for (int l = 1; l < (k - 1); l++)
                {
                    data[(k - l - 1) * dataSize + dataOffset] = div * ((dr + l) * data[(k - l - 2) * dataSize + dataOffset] + (k - l - dr) * data[(k - l - 1) * dataSize + dataOffset]);
                }
                data[0 * dataSize + dataOffset] = div * (1.0f - dr) * data[0 * dataSize + dataOffset];
            }

            const int particleWarpIndex = localIndexCalc % PARTICLES_PER_WARP;
            const int warpIndex = localIndexCalc / PARTICLES_PER_WARP; // should be just a real warp index!

            const int thetaGlobalOffsetBase = globalIndexBase * DIM * order;

            /* differentiate */
            // store dtheta to global
#if PME_GPU_PARALLEL_SPLINE
            k = orderIndex;
#else
#pragma unroll
            for (k = 0; k < order; k++)
#endif
            {
                const int thetaIndex = PME_SPLINE_THETA_STRIDE * (((k + order * warpIndex) * DIM + dimIndex) * PARTICLES_PER_WARP + particleWarpIndex);
                const int thetaGlobalIndex = thetaGlobalOffsetBase + thetaIndex;

                const real dtheta = ((k > 0) ? data[(k - 1) * dataSize + dataOffset] : 0.0f) - data[k * dataSize + dataOffset];
                dthetaGlobal[thetaGlobalIndex] = dtheta;
            }

            div             = 1.0f / (order - 1);
            data[(order - 1) * dataSize + dataOffset] = div * dr * data[(order - 2) * dataSize + dataOffset];
#pragma unroll
            for (int l = 1; l < (order - 1); l++)
            {
                data[(order - l - 1) * dataSize + dataOffset] = div * ((dr + l) * data[(order - l - 2) * dataSize + dataOffset] + (order - l - dr) * data[(order - l - 1) * dataSize + dataOffset]);
            }
            data[0 * dataSize + dataOffset] = div * (1.0f - dr) * data[0 * dataSize + dataOffset];

            // store theta to shared and global

#if PME_GPU_PARALLEL_SPLINE
            k = orderIndex;
#else
#pragma unroll
            for (k = 0; k < order; k++)
#endif
            {
                const int thetaIndex = PME_SPLINE_THETA_STRIDE * (((k + order * warpIndex) * DIM + dimIndex) * PARTICLES_PER_WARP + particleWarpIndex);
                const int thetaGlobalIndex = thetaGlobalOffsetBase + thetaIndex;

                theta[thetaIndex] = data[k * dataSize + dataOffset];
                thetaGlobal[thetaGlobalIndex] = data[k * dataSize + dataOffset];
            }
        }
    }
}

template <
        const int order,
        const int particlesPerBlock
        >
__device__ __forceinline__ void spread_charges(const real * __restrict__ coefficient,
                                              real * __restrict__ gridGlobal,
                                              const pme_gpu_const_parameters constants,
                                              const int globalIndex,
                                              const int localIndex,
                                              const int pny,
                                              const int pnz,
                                              const int * __restrict__ idx,
                                              const real * __restrict__ theta)
{
    /*
    pnx = pmegrid->s[XX];
    pny = pmegrid->s[YY];
    pnz = pmegrid->s[ZZ];

    offx = pmegrid->offset[XX];
    offy = pmegrid->offset[YY];
    offz = pmegrid->offset[ZZ];
    */
    const int offx = 0, offy = 0, offz = 0;
    // unused for now

    if ((globalIndex < constants.nAtoms) && (coefficient[localIndex] != 0.0f))
    {
        // spline Y/Z coordinates
        const int ithy = threadIdx.y;
        const int ithz = threadIdx.x; //?
        const int ix = idx[localIndex * DIM + XX] - offx;
        const int iy = idx[localIndex * DIM + YY] - offy;
        const int iz = idx[localIndex * DIM + ZZ] - offz;

        // copy
        const int particleWarpIndex = localIndex % PARTICLES_PER_WARP; // index of particle w.r.t. the warp (so, 0 or 1)
        const int warpIndex = localIndex / PARTICLES_PER_WARP; // should be just a normal warp index, actually!
        const int dimStride = PME_SPLINE_THETA_STRIDE * PARTICLES_PER_WARP;
        const int orderStride = dimStride * DIM;
        const int thetaOffsetBase = orderStride * order * warpIndex + particleWarpIndex;

        const real thz = theta[thetaOffsetBase + ithz * orderStride + ZZ * dimStride];
        const real thy = theta[thetaOffsetBase + ithy * orderStride + YY * dimStride];
        const real constVal = thz * thy * coefficient[localIndex];
        const int constOffset = (iy + ithy) * pnz + (iz + ithz);
        const real *thx = theta + (thetaOffsetBase + XX * dimStride);

#pragma unroll
        for (int ithx = 0; (ithx < order); ithx++)
        {
            const int index_x = (ix + ithx) * pny * pnz;
            atomicAdd(gridGlobal + index_x + constOffset, thx[ithx * orderStride] * constVal);
        }
    }
}

template <
        const int particlesPerBlock
        >
__device__ __forceinline__ void stage_charges(const int threadLocalId,
                                              real * __restrict__ coefficient,
                                              const real * __restrict__ coefficientGlobal)
{
    const int globalIndexBase = blockIdx.x * particlesPerBlock;
    if (threadLocalId < particlesPerBlock)
        coefficient[threadLocalId] = coefficientGlobal[globalIndexBase + threadLocalId];
}

template <
        const int particlesPerBlock
        >
__device__ __forceinline__ void stage_coordinates(const int threadLocalId,
                                              real * __restrict__ coordinates,
                                              const real * __restrict__ coordinatesGlobal)
{
    const int globalIndexBase = blockIdx.x * particlesPerBlock * DIM;
    const int index = threadLocalId - 1 * particlesPerBlock;
    if ((index >= 0) && (index < DIM * particlesPerBlock))
        coordinates[index] = coordinatesGlobal[globalIndexBase + index];
}

template <
        const int order,
        const int particlesPerBlock,
        const gmx_bool bCalcSplines, // first part
        const gmx_bool bCalcAlways,   // bypassing conditional in the first part
        const gmx_bool bSpread       // second part
        >
//#if GMX_PTX_ARCH <= 300
__launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
//#endif
//yupinov put bounds on separate kernels as well
__global__ void pme_spline_and_spread_kernel
(int start_ix, int start_iy, int start_iz,
 const int pny, const int pnz,
 const int3 nnOffset,
#if PME_USE_TEXTURES
#if USE_TEXOBJ
 cudaTextureObject_t nnTexture,
 cudaTextureObject_t fshTexture,
#endif
#else
 const int * __restrict__ nn,
 const real * __restrict__ fsh,
#endif
 const float3 * __restrict__ coordinatesGlobal,
 const real * __restrict__ coefficientGlobal,
 real * __restrict__ gridGlobal, real * __restrict__ thetaGlobal,
 real * __restrict__ dthetaGlobal, int * __restrict__ idxGlobal,
 const pme_gpu_const_parameters constants)
{
    // gridline indices
    __shared__ int idx[PME_SPREADGATHER_BLOCK_DATA_SIZE];
    // charges
    __shared__ real coefficient[particlesPerBlock];
    // spline parameters
    __shared__ real theta[PME_SPREADGATHER_BLOCK_DATA_SIZE * order];


    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;

    const int globalParticleIndexBase = blockIdx.x * particlesPerBlock;

    const int warpIndex = threadLocalId / warp_size;
    const int threadWarpIndex = threadLocalId % warp_size;
    const int particleWarpIndex = threadWarpIndex % PARTICLES_PER_WARP;
    const int localCalcIndex = warpIndex * PARTICLES_PER_WARP + particleWarpIndex;
    const int globalCalcIndex = globalParticleIndexBase + localCalcIndex;
    const int orderCalcIndex = threadWarpIndex / (PARTICLES_PER_WARP * DIM); // should be checked against order
    const int dimCalcIndex = (threadWarpIndex - orderCalcIndex * (PARTICLES_PER_WARP * DIM)) / PARTICLES_PER_WARP;

    if (bCalcSplines)
    {
        // coordinates
        __shared__ real coordinates[DIM * particlesPerBlock];

        stage_charges<particlesPerBlock>(threadLocalId, coefficient, coefficientGlobal);
        stage_coordinates<particlesPerBlock>(threadLocalId, coordinates, (const real *)coordinatesGlobal);
        __syncthreads();
        calculate_splines<order, particlesPerBlock, bCalcAlways>(nnOffset, (const float3 *)coordinates, coefficient,
                                                               thetaGlobal, theta, dthetaGlobal, idxGlobal, idx,
                                                               constants,
                                                               globalCalcIndex,
                                                               localCalcIndex,
                                                               globalParticleIndexBase,
                                                               dimCalcIndex,
                                                               orderCalcIndex);
    }
    else if (bSpread) // staging for spread
    {
        //yupinov - unmaintained
            /*
        if ((globalIndexCalc < n) && (dimIndex < DIM) && (localIndexCalc < particlesPerBlock))
        {
            idx[localIndexCalc * DIM + dimIndex] = idxGlobal[globalIndexCalc * DIM + dimIndex];

            const int thetaOffsetBase = localIndexCalc * DIM + dimIndex;
            const int thetaGlobalOffsetBase = globalIndexBase * DIM * order;
#pragma unroll
            for (int k = 0; k < order; k++)
            {
                const int thetaIndex = thetaOffsetBase + k * thetaStride;
                theta[thetaIndex] = thetaGlobal[thetaGlobalOffsetBase + thetaIndex];
            }
        }
        */
        stage_charges<particlesPerBlock>(threadLocalId, coefficient, coefficientGlobal);
        __syncthreads();
    }

    // SPREAD
    if (bSpread)
    {
        const int localSpreadIndex = threadIdx.z;
        const int globalSpreadIndex = globalParticleIndexBase + localSpreadIndex;
        spread_charges<order, particlesPerBlock>(coefficient, gridGlobal, constants, globalSpreadIndex, localSpreadIndex,
                                                pny, pnz, idx, theta);
    }
}


// pme_spline_and_spread split into pme_spline and pme_spread - as an experiment

template <
        const int order,
        const int particlesPerBlock,
        const gmx_bool bCalcAlways
        >
__global__ void pme_spline_kernel
(const int3 nnOffset,
#if PME_USE_TEXTURES
#if USE_TEXOBJ
  cudaTextureObject_t nnTexture,
  cudaTextureObject_t fshTexture,
#endif
#else
  const int * __restrict__ nn,
  const real * __restrict__ fsh,
#endif
 const float3 * __restrict__ coordinatesGlobal,
 const real * __restrict__ coefficientGlobal,
 real * __restrict__ thetaGlobal, real * __restrict__ dthetaGlobal, int * __restrict__ idxGlobal,
 const pme_gpu_const_parameters constants)
{
    // gridline indices
    __shared__ int idx[PME_SPREADGATHER_BLOCK_DATA_SIZE];
    // charges
    __shared__ real coefficient[particlesPerBlock];
    // coordinates
    __shared__ real coordinates[DIM * particlesPerBlock];
    // spline parameters
    __shared__ real theta[PME_SPREADGATHER_BLOCK_DATA_SIZE * order];

    const int globalIndexBase = blockIdx.x * particlesPerBlock;

    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;

    const int localIndexCalc = threadIdx.x;
    const int orderIndex = threadIdx.x; //yupinov - this is broken!
    const int dimIndex = threadIdx.y;
    const int globalIndexCalc = globalIndexBase + localIndexCalc;

    stage_charges<particlesPerBlock>(threadLocalId, coefficient, coefficientGlobal);
    stage_coordinates<particlesPerBlock>(threadLocalId, coordinates, (const real *)coordinatesGlobal);
    __syncthreads();

    calculate_splines<order, particlesPerBlock, bCalcAlways>(nnOffset, (const float3 *)coordinates, coefficient,
                                                           thetaGlobal, theta, dthetaGlobal, idxGlobal, idx,
                                                           constants,
                                                           globalIndexCalc,
                                                           localIndexCalc,
                                                           globalIndexBase,
                                                           dimIndex,
                                                           orderIndex);
}


template
<const int order, const int particlesPerBlock>
__global__ void pme_spread_kernel
( //int start_ix, int start_iy, int start_iz,
  const int pny, const int pnz,
 const real * __restrict__ coefficientGlobal,
 real * __restrict__ gridGlobal, real * __restrict__ thetaGlobal, const int * __restrict__ idxGlobal,
             const pme_gpu_const_parameters constants)
{
    __shared__ int idx[PME_SPREADGATHER_BLOCK_DATA_SIZE];
    __shared__ real coefficient[particlesPerBlock];

    __shared__ real theta[PME_SPREADGATHER_BLOCK_DATA_SIZE * order];

    const int localIndex = threadIdx.x;
    const int globalParticleIndexBase = blockIdx.x * particlesPerBlock;
    const int globalIndex = globalParticleIndexBase + localIndex;


    //yupinov - staging
    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;

    stage_charges<particlesPerBlock>(threadLocalId, coefficient, coefficientGlobal);
    __syncthreads();

    const int localIndexCalc = threadLocalId / DIM;
    const int dimIndex = threadLocalId - localIndexCalc * DIM;
    const int globalIndexCalc = globalParticleIndexBase + localIndexCalc;

    if ((globalIndexCalc < constants.nAtoms) && (dimIndex < DIM) && (localIndexCalc < particlesPerBlock))
    {
        idx[localIndexCalc * DIM + dimIndex] = idxGlobal[globalIndexCalc * DIM + dimIndex];

        //unmaintained...
        const int thetaOffsetBase = localIndexCalc * DIM + dimIndex;
        const int thetaGlobalOffsetBase = globalParticleIndexBase * DIM * order;
#pragma unroll
        for (int k = 0; k < order; k++)
        {
            const int thetaIndex = thetaOffsetBase + k * PME_SPLINE_ORDER_STRIDE;
            theta[thetaIndex] = thetaGlobal[thetaGlobalOffsetBase + thetaIndex];
        }
    }
    __syncthreads();

    // SPREAD
    spread_charges<order, particlesPerBlock>(coefficient, gridGlobal, constants, globalIndex, localIndex,
                                            pny, pnz, idx, theta);
}

template <
    const int order
    >
__global__ void pme_wrap_kernel
    (const int nx, const int ny, const int nz,
     const int pny, const int pnz,
    const pme_gpu_overlap_t OVERLAP,
     real * __restrict__ grid
     )
{
    const int blockId = blockIdx.x
                 + blockIdx.y * gridDim.x
                 + gridDim.x * gridDim.y * blockIdx.z;
    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;
    const int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadLocalId;

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
        const int ixy = cellIndex / zoneSizeYZ.y; //yupinov check expensive integer divisions everywhere!
        const int iz = cellIndex - zoneSizeYZ.y * ixy;
        const int ix = ixy / zoneSizeYZ.x;
        const int iy = ixy - zoneSizeYZ.x * ix;
        const int targetIndex = (ix * pny + iy) * pnz + iz;

        int sourceOffset = 0;

        // stage those bits in constant memory as well
        const int overlapZ = ((zoneIndex == 0) || (zoneIndex == 3) || (zoneIndex == 4) || (zoneIndex == 6)) ? 1 : 0;
        const int overlapY = ((zoneIndex == 1) || (zoneIndex == 3) || (zoneIndex == 5) || (zoneIndex == 6)) ? 1 : 0;
        const int overlapX = ((zoneIndex == 2) || (zoneIndex > 3)) ? 1 : 0;
        if (overlapZ)
        {
            sourceOffset = nz;
        }
        if (overlapY)
        {
            sourceOffset += ny * pnz;
        }
        if (overlapX)
        {
            sourceOffset += nx * pny * pnz;
        }
        const int sourceIndex = targetIndex + sourceOffset;

        /* // condition for atomic seems a bit excessive - test on different hardware?
        const int targetOverlapX = (ix < overlap) ? 1 : 0;
        const int targetOverlapY = (iy < overlap) ? 1 : 0;
        const int targetOverlapZ = (iz < overlap) ? 1 : 0;
        const int useAtomic = ((targetOverlapX + targetOverlapY + targetOverlapZ) > 1) ? 1 : 0;
        */
        const int useAtomic = 1;
        if (useAtomic)
            atomicAdd(grid + targetIndex, grid[sourceIndex]);
        else
            grid[targetIndex] += grid[sourceIndex];
    }
}

void pme_gpu_copy_calcspline_constants(gmx_pme_t *pme)
{
    cudaStream_t s = pme->gpu->pmeStream;

    cudaError_t stat;

    const int nx = pme->nkx;
    const int ny = pme->nky;
    const int nz = pme->nkz;

    const int fshSize = 5 * (nx + ny + nz) * sizeof(real);
    real *fshArray = pme->gpu->fshArray = (real *)PMEMemoryFetch(pme, PME_ID_FSH, fshSize, ML_DEVICE);
    cu_copy_H2D_async(fshArray                , pme->fshx, 5 * nx * sizeof(real), s);
    cu_copy_H2D_async(fshArray + 5 * nx       , pme->fshy, 5 * ny * sizeof(real), s);
    cu_copy_H2D_async(fshArray + 5 * (nx + ny), pme->fshz, 5 * nz * sizeof(real), s);

    const int nnSize = 5 * (nx + ny + nz) * sizeof(int);
    int *nnArray = pme->gpu->nnArray = (int *)PMEMemoryFetch(pme, PME_ID_NN, nnSize, ML_DEVICE);
    cu_copy_H2D_async(nnArray                , pme->nnx, 5 * nx * sizeof(int), s);
    cu_copy_H2D_async(nnArray + 5 * nx       , pme->nny, 5 * ny * sizeof(int), s);
    cu_copy_H2D_async(nnArray + 5 * (nx + ny), pme->nnz, 5 * nz * sizeof(int), s);

#if PME_USE_TEXTURES
#if USE_TEXOBJ
    //if (use_texobj(dev_info))
    // should check device info here for CC >= 3.0
    {
        cudaResourceDesc rd;
        cudaTextureDesc td;

        memset(&rd, 0, sizeof(rd));
        rd.resType                  = cudaResourceTypeLinear;
        rd.res.linear.devPtr        = fshArray;
        rd.res.linear.desc.f        = cudaChannelFormatKindFloat;
        rd.res.linear.desc.x        = 32;
        rd.res.linear.sizeInBytes   = fshSize;
        memset(&td, 0, sizeof(td));
        td.readMode                 = cudaReadModeElementType;
        stat = cudaCreateTextureObject(&fshTexture, &rd, &td, NULL);
        CU_RET_ERR(stat, "cudaCreateTextureObject on fsh_d failed");


        memset(&rd, 0, sizeof(rd));
        rd.resType                  = cudaResourceTypeLinear;
        rd.res.linear.devPtr        = nnArray;
        rd.res.linear.desc.f        = cudaChannelFormatKindSigned;
        rd.res.linear.desc.x        = 32;
        rd.res.linear.sizeInBytes   = nnSize;
        memset(&td, 0, sizeof(td));
        td.readMode                 = cudaReadModeElementType;
        stat = cudaCreateTextureObject(&nnTexture, &rd, &td, NULL); //yupinov destroy, keep allocated
        CU_RET_ERR(stat, "cudaCreateTextureObject on nn_d failed");
    }
    //else
#else
    {
        cudaChannelFormatDesc cd_fsh = cudaCreateChannelDesc<float>();
        stat = cudaBindTexture(NULL, &fshTextureRef, fshArray, &cd_fsh, fshSize);
        CU_RET_ERR(stat, "cudaBindTexture on fsh failed");

        cudaChannelFormatDesc cd_nn = cudaCreateChannelDesc<int>();
        stat = cudaBindTexture(NULL, &nnTextureRef, nnArray, &cd_nn, nnSize);
        CU_RET_ERR(stat, "cudaBindTexture on nn failed");
        //yupinov unbind
    }
#endif
#endif
}

void pme_gpu_alloc_grids(gmx_pme_t *pme, const int gmx_unused grid_index)
{
    const int pnx = pme->pmegrid_nx;
    const int pny = pme->pmegrid_ny;
    const int pnz = pme->pmegrid_nz;
    const int gridSize = pnx * pny * pnz * sizeof(real);

    pme->gpu->grid = (real *)PMEMemoryFetch(pme, PME_ID_REAL_GRID, gridSize, ML_DEVICE);
    if (pme->gpu->bOutOfPlaceFFT)
        pme->gpu->fourierGrid = (t_complex *)PMEMemoryFetch(pme, PME_ID_COMPLEX_GRID, gridSize, ML_DEVICE);
    else
        pme->gpu->fourierGrid = (t_complex *)pme->gpu->grid;
}

void pme_gpu_clear_grid(gmx_pme_t *pme, const int gmx_unused grid_index)
{
    /*
    pmegrid_t *pmegrid = &(pme->pmegrid[grid_index].grid);
    const int pnx = pmegrid->n[XX];
    const int pny = pmegrid->n[YY];
    const int pnz = pmegrid->n[ZZ];
    */

    const int pnx = pme->pmegrid_nx;
    const int pny = pme->pmegrid_ny;
    const int pnz = pme->pmegrid_nz;
    const int gridSize = pnx * pny * pnz * sizeof(real);

    cudaStream_t s = pme->gpu->pmeStream;

    cudaError_t stat = cudaMemsetAsync(pme->gpu->grid, 0, gridSize, s);
    CU_RET_ERR(stat, "cudaMemsetAsync spread error");
}

void spread_on_grid_gpu(gmx_pme_t *pme, pme_atomcomm_t *atc,
         const int gmx_unused grid_index,
         pmegrid_t *pmegrid,
         const gmx_bool bCalcSplines,
         const gmx_bool bSpread,
         const gmx_bool bDoSplines)
{
    const gmx_bool bSeparateKernels = false;  // significantly slower if true
    if (!bCalcSplines && !bSpread)
        gmx_fatal(FARGS, "No splining or spreading to be done?"); //yupinov use of gmx_fatal

    //yupinov
    // bCalcSplines is always true - untested, unfinished
    // bDoSplines is always false - untested
    // bSpread is always true - untested, unfinished

    cudaStream_t s = pme->gpu->pmeStream;

    //int nx = pmegrid->s[XX], ny = pmegrid->s[YY], nz = pmegrid->s[ZZ];
    const int order = pmegrid->order;
    const int overlap = order - 1;

    /*
    ivec local_ndata, local_size, local_offset;
    gmx_parallel_3dfft_real_limits_gpu(pme, grid_index, local_ndata, local_offset, local_size);
    const int pnx = local_size[XX];
    const int pny = local_size[YY];
    const int pnz = local_size[ZZ];
    const int nx = local_ndata[XX];
    const int ny = local_ndata[YY];
    const int nz = local_ndata[ZZ];
    */
    const int pnx = pmegrid->n[XX];
    const int pny = pmegrid->n[YY];
    const int pnz = pmegrid->n[ZZ];
    const int nx = pme->nkx;
    const int ny = pme->nky;
    const int nz = pme->nkz;

    const int n = pme->gpu->constants.nAtoms;

    const int gridSize = pnx * pny * pnz * sizeof(real);

    int size_order = order * n * sizeof(real);
    int size_order_dim = size_order * DIM;
    real *theta_d = (real *)PMEMemoryFetch(pme, PME_ID_THETA, size_order_dim, ML_DEVICE);
    real *dtheta_d = (real *)PMEMemoryFetch(pme, PME_ID_DTHETA, size_order_dim, ML_DEVICE);

    // IDXPTR
    int idx_size = n * DIM * sizeof(int);
    int *idx_d = (int *)PMEMemoryFetch(pme, PME_ID_IDXPTR, idx_size, ML_DEVICE);

    const int3 nnOffset = {0, 5 * nx, 5 * (nx + ny)};


    if (bCalcSplines)
    {
         /*
        const size_t coordinatesSize = DIM * n_blocked * sizeof(real);
        float3 *xptr_h = (float3 *)PMEMemoryFetch(pme, PME_ID_XPTR, coordinatesSize, ML_HOST);
        memcpy(xptr_h, atc->x, coordinatesSize);
        xptr_d = (float3 *)PMEMemoryFetch(pme, PME_ID_XPTR, coordinatesSize, ML_DEVICE);
        cu_copy_H2D_async(xptr_d, xptr_h, coordinatesSize, ML_DEVICE, pme->gpu->pmeStream);
        */
    }

    // each spread kernel thread works on [order] contiguous x grid points, so we multiply the total number of threads by [order^2]
    // so only [1/order^2] of all kernel threads works on particle splines -> does it make sense to split it like this

    const int blockSize = THREADS_PER_BLOCK;
    const int particlesPerBlock = blockSize / order / order;
    const int splineParticlesPerBlock = particlesPerBlock; //blockSize / DIM; - can be easily changed, just have to pass spread theta stride to the spline kernel!
    // duplicated below!

    dim3 nBlocksSpread((n + blockSize - 1) / blockSize * order * order);
    dim3 nBlocksSpline((n + splineParticlesPerBlock - 1) / splineParticlesPerBlock);
    dim3 dimBlockSpread(order, order, particlesPerBlock); // used for spline_and_spread / spread
    dim3 dimBlockSpline(splineParticlesPerBlock, DIM); // used for spline
    switch (order)
    {
        case 4:
            if (bSeparateKernels)
            {
                if (bCalcSplines)
                {
                    pme_gpu_timing_start(pme, ewcsPME_SPLINE);

                    if (bDoSplines)
                        gmx_fatal(FARGS, "the code for bDoSplines==true was not tested!");
                    else
                    {
                        pme_spline_kernel<4, blockSize / 4 / 4, FALSE> <<<nBlocksSpline, dimBlockSpline, 0, s>>>
                                                                                                   (nnOffset,
#if PME_USE_TEXTURES
#if USE_TEXOBJ
                                                                                                    nnTexture, fshTexture,
#endif
#else
                                                                                                    pme->gpu->nnArray, pme->gpu->fshArray,
#endif
                                                                                                    pme->gpu->coordinates,
                                                                                                    pme->gpu->coefficients,
                                                                                                    theta_d, dtheta_d, idx_d,
                                                                                                    pme->gpu->constants);


                    }

                    CU_LAUNCH_ERR("pme_spline_kernel");

                    pme_gpu_timing_stop(pme, ewcsPME_SPLINE);
                }
                if (bSpread)
                {
                    pme_gpu_timing_start(pme, ewcsPME_SPREAD);

                    pme_spread_kernel<4, blockSize / 4 / 4> <<<nBlocksSpread, dimBlockSpread, 0, s>>>
                                                                            (/*pme->pmegrid_start_ix, pme->pmegrid_start_iy, pme->pmegrid_start_iz,*/
                                                                             pny, pnz,
                                                                             pme->gpu->coefficients,
                                                                             pme->gpu->grid, theta_d, idx_d,
                                                                             pme->gpu->constants);

                    CU_LAUNCH_ERR("pme_spread_kernel");

                    pme_gpu_timing_stop(pme, ewcsPME_SPREAD);
                }
            }
            else // a single monster kernel here
            {
                pme_gpu_timing_start(pme, ewcsPME_SPLINEANDSPREAD);

                if (bCalcSplines)
                {
                    if (bDoSplines)
                        gmx_fatal(FARGS, "the code for bDoSplines==true was not tested!");
                    else
                    {
                        if (bSpread)
                        {
                            pme_spline_and_spread_kernel<4, blockSize / 4 / 4, TRUE, FALSE, TRUE> <<<nBlocksSpread, dimBlockSpread, 0, s>>>
                                  (pme->pmegrid_start_ix, pme->pmegrid_start_iy, pme->pmegrid_start_iz,
                                   pny, pnz,
                                   nnOffset,
#if PME_USE_TEXTURES
#if USE_TEXOBJ
                                   nnTexture, fshTexture,
#endif
#else
                                   pme->gpu->nnArray, pme->gpu->fshArray,
#endif
                                   pme->gpu->coordinates, pme->gpu->coefficients, pme->gpu->grid, theta_d, dtheta_d, idx_d,
                                   pme->gpu->constants);
                        }
                        else
                            gmx_fatal(FARGS, "the code for bSpread==false was not tested!");
                    }
                }
                else
                    gmx_fatal(FARGS, "the code for bCalcSplines==false was not tested!");
                CU_LAUNCH_ERR("pme_spline_and_spread_kernel");

                pme_gpu_timing_stop(pme, ewcsPME_SPLINEANDSPREAD);
            }
            if (bSpread && pme->gpu->bGPUSingle)
            {
                // wrap on GPU as a separate small kernel - we need a complete grid first!
                const int blockSize = 4 * warp_size; //yupinov this is everywhere! and architecture-specific
                const int overlappedCells = (nx + overlap) * (ny + overlap) * (nz + overlap) - nx * ny * nz;
                const int nBlocks = (overlappedCells + blockSize - 1) / blockSize;

                pme_gpu_timing_start(pme, ewcsPME_WRAP);

                pme_wrap_kernel<4> <<<nBlocks, blockSize, 0, s>>>(nx, ny, nz, pny, pnz,
                                                                  pme->gpu->overlap,
                                                                  pme->gpu->grid);

                CU_LAUNCH_ERR("pme_wrap_kernel");

                pme_gpu_timing_stop(pme, ewcsPME_WRAP);
            }
            break;

        default:
            gmx_fatal(FARGS, "the code for pme_order != 4 was not tested!");
    }

    if (!pme->gpu->bGPUFFT && bSpread)
    {
        cu_copy_D2H_async(pmegrid->grid, pme->gpu->grid, gridSize, s);
        cudaError_t stat = cudaEventRecord(pme->gpu->syncSpreadGridD2H, s);
        CU_RET_ERR(stat, "PME spread grid sync fail");
    }
    if (!pme->gpu->bGPUGather)
    {
        //yupinov - (d)theta layout is not straightforward on GPU, would fail with CPU gather
        // and no accounting for PME communication (bGPUSingle check?)
        for (int j = 0; j < DIM; ++j)
        {
            cu_copy_D2H_async(atc->spline[0].dtheta[j], dtheta_d + j * n * order, size_order, s);
            cu_copy_D2H_async(atc->spline[0].theta[j], theta_d + j * n * order, size_order, s);
        }
        cu_copy_D2H_async(atc->idx, idx_d, idx_size, s);
    }
}

