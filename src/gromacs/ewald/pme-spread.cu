/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013-2016, by the GROMACS development team, led by
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
 *  \brief Implements PME GPU spline calculation and charge spreading in CUDA.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include "pme.h"
#include "pme-internal.h"

#include "gromacs/utility/fatalerror.h"

#include "pme.cuh"

#define PME_GPU_PARALLEL_SPLINE 1
/* This define affects the spline calculation behaviour in the spreading kernel.
 * 0: a single GPU thread handles a single dimension of a single particle (calculating and storing (order) spline values and derivatives).
 * 1: (order) threads do redundant work on this same task, each one stores only a single theta and single dtheta into global arrays.
 * The only efficiency difference is less global store operations, countered by more redundant spline computation.
 */

#define THREADS_PER_BLOCK   (4 * warp_size)
#define MIN_BLOCKS_PER_MP   (16)

#if PME_USE_TEXTURES
texture<int, 1, cudaReadModeElementType>   nnTextureRef;
texture<float, 1, cudaReadModeElementType> fshTextureRef;
#endif


/* This is the PME GPU spline calculation .
 * It corresponds to the CPU codepath functions calc_interpolation_idx and make_bsplines.
 */
template <
    const int order,
    const int particlesPerBlock,
    const gmx_bool bCalcAlways
    >
__device__ __forceinline__ void calculate_splines(const float3 * __restrict__     coordinates,
                                                  float * __restrict__            coefficient,
                                                  float * __restrict__            theta,
                                                  int * __restrict__              gridlineIndices,
                                                  const pme_gpu_kernel_params     kernelParams,
                                                  const int                       globalIndexCalc,
                                                  const int                       localIndexCalc,
                                                  const int                       globalIndexBase,
                                                  const int                       dimIndex,
                                                  const int                       orderIndex)
{
    /* Global memory pointers */
    float * __restrict__ thetaGlobal           = kernelParams.atoms.theta;
    float * __restrict__ dthetaGlobal          = kernelParams.atoms.dtheta;
    int * __restrict__   gridlineIndicesGlobal = kernelParams.atoms.gridlineIndices;

    /* Fractional coordinates */
    __shared__ float fractX[PME_SPREADGATHER_BLOCK_DATA_SIZE];

    const int        sharedMemoryIndex = localIndexCalc * DIM + dimIndex;

    const int        dataSize   = PME_GPU_PARALLEL_SPLINE ? PME_SPREADGATHER_BLOCK_DATA_SIZE : 1;
    const int        dataOffset = PME_GPU_PARALLEL_SPLINE ? sharedMemoryIndex : 0;

    /* Spline parameter storage, shared for PME_GPU_PARALLEL_SPLINE==1 to not overuse the local memory */
#if PME_GPU_PARALLEL_SPLINE
    __shared__
#endif
    float data[dataSize * order];

    const int localCheck  = (dimIndex < DIM) && (orderIndex < (PME_GPU_PARALLEL_SPLINE ? order : 1));
    const int globalCheck = pme_gpu_check_atom_data_index(globalIndexCalc, kernelParams.atoms.nAtoms);

    if (localCheck && globalCheck)
    {
        /* Indices interpolation */

        if (orderIndex == 0)
        {
            int           fShiftIndex, tInt;
            float         n, t;
            const float3  x = coordinates[localIndexCalc];
            /* Accessing fields in fshOffset/nXYZ/recipbox/... with dimIndex offset
             * puts them into local memory(!) instead of accessing the constant memory directly.
             * That's the reason for the switch, to unroll explicitly.
             * The commented parts correspond to the 0 components of the recipbox.
             */
            switch (dimIndex)
            {
                case 0:
                    fShiftIndex = kernelParams.grid.fshOffset.x;
                    n           = kernelParams.grid.localGridSizeFP.x;
                    t           = x.x * kernelParams.step.recipbox[dimIndex].x + x.y * kernelParams.step.recipbox[dimIndex].y + x.z * kernelParams.step.recipbox[dimIndex].z;
                    break;

                case 1:
                    fShiftIndex = kernelParams.grid.fshOffset.y;
                    n           = kernelParams.grid.localGridSizeFP.y;
                    t           = /*x.x * constants.step.recipbox[dimIndex].x + */ x.y * kernelParams.step.recipbox[dimIndex].y + x.z * kernelParams.step.recipbox[dimIndex].z;
                    break;

                case 2:
                    fShiftIndex = kernelParams.grid.fshOffset.z;
                    n           = kernelParams.grid.localGridSizeFP.z;
                    t           = /*x.x * constants.step.recipbox[dimIndex].x + x.y * constants.step.recipbox[dimIndex].y + */ x.z * kernelParams.step.recipbox[dimIndex].z;
                    break;
            }

            /* Fractional coordinates along box vectors, adding 2.0 to make 100% sure we are positive for triclinic boxes */
            t    = (t + 2.0f) * n;
            tInt = (int)t;
            fractX[sharedMemoryIndex] = t - tInt;
            fShiftIndex              += tInt;

#if PME_USE_TEXTURES
#if PME_USE_TEXOBJ
            fractX[sharedMemoryIndex]             += tex1Dfetch<float>(kernelParams.grid.fshTexture, fShiftIndex);
            gridlineIndices[sharedMemoryIndex]     = tex1Dfetch<int>(kernelParams.grid.nnTexture, fShiftIndex);
#else
            fractX[sharedMemoryIndex]             += tex1Dfetch(fshTextureRef, fShiftIndex);
            gridlineIndices[sharedMemoryIndex]     = tex1Dfetch(nnTextureRef, fShiftIndex);
#endif
#else
            const float * __restrict__  fsh = constants.grid.fshArray;
            const int * __restrict__    nn  = constants.grid.nnArray;
            fractX[sharedMemoryIndex]         += fsh[fShiftIndex];
            gridlineIndices[sharedMemoryIndex] = nn[fShiftIndex];
#endif

            gridlineIndicesGlobal[globalIndexBase * DIM + sharedMemoryIndex] = gridlineIndices[sharedMemoryIndex];
        }

        /* B-spline calculation */

        if (bCalcAlways || (coefficient[localIndexCalc] != 0.0f)) //yupinov thsi conditional should be same in spread and gather!
        {
            float       div;
            int         k;

            const float dr = fractX[sharedMemoryIndex];

            /* dr is relative offset from lower cell limit */
            data[(order - 1) * dataSize + dataOffset] = 0.0f;
            data[1 * dataSize + dataOffset]           = dr;
            data[0 * dataSize + dataOffset]           = 1.0f - dr;

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

            const int particleWarpIndex = localIndexCalc % PME_SPREADGATHER_PARTICLES_PER_WARP;
            const int warpIndex         = localIndexCalc / PME_SPREADGATHER_PARTICLES_PER_WARP;

            const int thetaGlobalOffsetBase = globalIndexBase * DIM * order;

            /* Differentiation and storing the spline derivatives (dtheta) */
#if PME_GPU_PARALLEL_SPLINE
            k = orderIndex;
#else
#pragma unroll
            for (k = 0; k < order; k++)
#endif
            {
                const int   thetaIndex       = PME_SPLINE_THETA_STRIDE * (((k + order * warpIndex) * DIM + dimIndex) * PME_SPREADGATHER_PARTICLES_PER_WARP + particleWarpIndex);
                const int   thetaGlobalIndex = thetaGlobalOffsetBase + thetaIndex;

                const float dtheta = ((k > 0) ? data[(k - 1) * dataSize + dataOffset] : 0.0f) - data[k * dataSize + dataOffset];
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

            /* Storing the spline values (theta) */
#if PME_GPU_PARALLEL_SPLINE
            k = orderIndex;
#else
#pragma unroll
            for (k = 0; k < order; k++)
#endif
            {
                const int thetaIndex       = PME_SPLINE_THETA_STRIDE * (((k + order * warpIndex) * DIM + dimIndex) * PME_SPREADGATHER_PARTICLES_PER_WARP + particleWarpIndex);
                const int thetaGlobalIndex = thetaGlobalOffsetBase + thetaIndex;

                theta[thetaIndex]             = data[k * dataSize + dataOffset];
                thetaGlobal[thetaGlobalIndex] = data[k * dataSize + dataOffset];
            }
        }
    }
}

template <
    const int order,
    const int particlesPerBlock
    >
__device__ __forceinline__ void spread_charges(const float * __restrict__      coefficient,
                                               const pme_gpu_kernel_params     kernelParams,
                                               const int                       globalIndex,
                                               const int                       localIndex,
                                               const int * __restrict__        gridlineIndices,
                                               const float * __restrict__      theta)
{
    /* Global memory pointer */
    float * __restrict__ gridGlobal = kernelParams.grid.realGrid;

    const int            pny        = kernelParams.grid.localGridSizePadded.y;
    const int            pnz        = kernelParams.grid.localGridSizePadded.z;

    const int            offx = 0, offy = 0, offz = 0;
    // unused for now

    const int globalCheck = pme_gpu_check_atom_data_index(globalIndex, kernelParams.atoms.nAtoms);
    if ((coefficient[localIndex] != 0.0f) && globalCheck)
    {
        // spline Y/Z coordinates
        const int ithy = threadIdx.y;
        const int ithz = threadIdx.x; //?
        const int ix   = gridlineIndices[localIndex * DIM + XX] - offx;
        const int iy   = gridlineIndices[localIndex * DIM + YY] - offy;
        const int iz   = gridlineIndices[localIndex * DIM + ZZ] - offz;

        // copy
        const int    particleWarpIndex = localIndex % PME_SPREADGATHER_PARTICLES_PER_WARP; // index of particle w.r.t. the warp (so, 0 or 1)
        const int    warpIndex         = localIndex / PME_SPREADGATHER_PARTICLES_PER_WARP; // should be just a normal warp index, actually!
        const int    dimStride         = PME_SPLINE_THETA_STRIDE * PME_SPREADGATHER_PARTICLES_PER_WARP;
        const int    orderStride       = dimStride * DIM;
        const int    thetaOffsetBase   = orderStride * order * warpIndex + particleWarpIndex;

        const float  thz         = theta[thetaOffsetBase + ithz * orderStride + ZZ * dimStride];
        const float  thy         = theta[thetaOffsetBase + ithy * orderStride + YY * dimStride];
        const float  constVal    = thz * thy * coefficient[localIndex];
        const int    constOffset = (iy + ithy) * pnz + (iz + ithz);
        const float *thx         = theta + (thetaOffsetBase + XX * dimStride);

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
__device__ __forceinline__ void stage_charges(const int                  threadLocalId,
                                              float * __restrict__       coefficient,
                                              const float * __restrict__ coefficientGlobal,
                                              const int                  nAtoms)
{
    const int globalIndexBase = blockIdx.x * particlesPerBlock;
    const int index           = threadLocalId;
    const int globalIndex     = globalIndexBase + index;
    const int globalCheck     = pme_gpu_check_atom_data_index(globalIndex, nAtoms);
    if ((index < particlesPerBlock) && globalCheck)
    {
        coefficient[threadLocalId] = coefficientGlobal[globalIndex];
    }
}

template <
    const int particlesPerBlock
    >
__device__ __forceinline__ void stage_coordinates(const int                  threadLocalId,
                                                  float * __restrict__       coordinates,
                                                  const float * __restrict__ coordinatesGlobal,
                                                  const int                  nAtoms)
{
    const int globalIndexBase = blockIdx.x * particlesPerBlock * DIM;
    const int index           = threadLocalId - 1 * particlesPerBlock;
    const int globalIndex     = globalIndexBase + index;
    const int globalCheck     = pme_gpu_check_atom_data_index(globalIndex, DIM * nAtoms); /* DIM floats per atom */
    if ((index >= 0) && (index < DIM * particlesPerBlock) && globalCheck)
    {
        coordinates[index] = coordinatesGlobal[globalIndex];
    }
}

template <
    const int order,
    const int particlesPerBlock,
    const gmx_bool bCalcSplines,     // first part
    const gmx_bool bCalcAlways,      // bypassing conditional in the first part
    const gmx_bool bSpread           // second part
    >
//#if GMX_PTX_ARCH <= 300
__launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
//#endif
//yupinov put bounds on separate kernels as well
__global__ void pme_spline_and_spread_kernel(const pme_gpu_kernel_params kernelParams)
{
    /* Global memory pointers */
    const float3 * __restrict__  coordinatesGlobal = kernelParams.atoms.coordinates;
    const float * __restrict__   coefficientGlobal = kernelParams.atoms.coefficients;

    /* Gridline indices, ivec */
    __shared__ int                gridlineIndices[PME_SPREADGATHER_BLOCK_DATA_SIZE];
    // charges
    __shared__ float              coefficient[particlesPerBlock];
    // spline parameters
    __shared__ float              theta[PME_SPREADGATHER_BLOCK_DATA_SIZE * order];

    const int                     threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;

    const int globalParticleIndexBase = blockIdx.x * particlesPerBlock;

    const int warpIndex         = threadLocalId / warp_size;
    const int threadWarpIndex   = threadLocalId % warp_size;
    const int particleWarpIndex = threadWarpIndex % PME_SPREADGATHER_PARTICLES_PER_WARP;
    const int localCalcIndex    = warpIndex * PME_SPREADGATHER_PARTICLES_PER_WARP + particleWarpIndex;
    const int globalCalcIndex   = globalParticleIndexBase + localCalcIndex;
    const int orderCalcIndex    = threadWarpIndex / (PME_SPREADGATHER_PARTICLES_PER_WARP * DIM); // should be checked against order
    const int dimCalcIndex      = (threadWarpIndex - orderCalcIndex * (PME_SPREADGATHER_PARTICLES_PER_WARP * DIM)) / PME_SPREADGATHER_PARTICLES_PER_WARP;

    if (bCalcSplines)
    {
        // coordinates
        __shared__ float coordinates[DIM * particlesPerBlock];

        stage_charges<particlesPerBlock>(threadLocalId, coefficient, coefficientGlobal, kernelParams.atoms.nAtoms);
        stage_coordinates<particlesPerBlock>(threadLocalId, coordinates, (const float *)coordinatesGlobal, kernelParams.atoms.nAtoms);
        __syncthreads();
        calculate_splines<order, particlesPerBlock, bCalcAlways>((const float3 *)coordinates, coefficient,
                                                                 theta, gridlineIndices,
                                                                 kernelParams,
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
           gridlineIndices[localIndexCalc * DIM + dimIndex] = gridlineIndicesGlobal[globalIndexCalc * DIM + dimIndex];

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
        stage_charges<particlesPerBlock>(threadLocalId, coefficient, coefficientGlobal, kernelParams.atoms.nAtoms);
        __syncthreads();
    }

    // SPREAD
    if (bSpread)
    {
        const int localSpreadIndex  = threadIdx.z;
        const int globalSpreadIndex = globalParticleIndexBase + localSpreadIndex;
        spread_charges<order, particlesPerBlock>(coefficient, kernelParams, globalSpreadIndex, localSpreadIndex,
                                                 gridlineIndices, theta);
    }
}


// pme_spline_and_spread split into pme_spline and pme_spread - as an experiment

template <
    const int order,
    const int particlesPerBlock,
    const gmx_bool bCalcAlways
    >
__global__ void pme_spline_kernel(const pme_gpu_kernel_params kernelParams)
{
    /* Global memory pointers */
    const float3 * __restrict__  coordinatesGlobal = kernelParams.atoms.coordinates;
    const float * __restrict__   coefficientGlobal = kernelParams.atoms.coefficients;

    // gridline indices
    __shared__ int               gridlineIndices[PME_SPREADGATHER_BLOCK_DATA_SIZE];
    // charges
    __shared__ float             coefficient[particlesPerBlock];
    // coordinates
    __shared__ float             coordinates[DIM * particlesPerBlock];
    // spline parameters
    __shared__ float             theta[PME_SPREADGATHER_BLOCK_DATA_SIZE * order];

    const int                    globalIndexBase = blockIdx.x * particlesPerBlock;

    const int                    threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;

    const int localIndexCalc  = threadIdx.x;
    const int orderIndex      = threadIdx.x; //yupinov - this is broken!
    const int dimIndex        = threadIdx.y;
    const int globalIndexCalc = globalIndexBase + localIndexCalc;

    stage_charges<particlesPerBlock>(threadLocalId, coefficient, coefficientGlobal, kernelParams.atoms.nAtoms);
    stage_coordinates<particlesPerBlock>(threadLocalId, coordinates, (const float *)coordinatesGlobal, kernelParams.atoms.nAtoms);
    __syncthreads();
    // TODO: clean up type casts

    calculate_splines<order, particlesPerBlock, bCalcAlways>((const float3 *)coordinates, coefficient,
                                                             theta, gridlineIndices,
                                                             kernelParams,
                                                             globalIndexCalc,
                                                             localIndexCalc,
                                                             globalIndexBase,
                                                             dimIndex,
                                                             orderIndex);
}


template
<const int order, const int particlesPerBlock>
__global__ void pme_spread_kernel(const pme_gpu_kernel_params kernelParams)
{
    /* Global memory pointers */
    const float * __restrict__ coefficientGlobal     = kernelParams.atoms.coefficients;
    const int * __restrict__   gridlineIndicesGlobal = kernelParams.atoms.gridlineIndices;
    float * __restrict__       thetaGlobal           = kernelParams.atoms.theta;


    __shared__ int       gridlineIndices[PME_SPREADGATHER_BLOCK_DATA_SIZE];
    __shared__ float     coefficient[particlesPerBlock];

    __shared__ float     theta[PME_SPREADGATHER_BLOCK_DATA_SIZE * order];

    const int            localIndex              = threadIdx.x;
    const int            globalParticleIndexBase = blockIdx.x * particlesPerBlock;
    const int            globalIndex             = globalParticleIndexBase + localIndex;


    //yupinov - staging
    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;


    stage_charges<particlesPerBlock>(threadLocalId, coefficient, coefficientGlobal, kernelParams.atoms.nAtoms);
    __syncthreads();

    const int localIndexCalc  = threadLocalId / DIM;
    const int dimIndex        = threadLocalId - localIndexCalc * DIM;
    const int globalIndexCalc = globalParticleIndexBase + localIndexCalc;
    const int globalCheck     = pme_gpu_check_atom_data_index(globalIndexCalc, kernelParams.atoms.nAtoms);

    if ((dimIndex < DIM) && (localIndexCalc < particlesPerBlock) && globalCheck)
    {
        gridlineIndices[localIndexCalc * DIM + dimIndex] = gridlineIndicesGlobal[globalIndexCalc * DIM + dimIndex];

        //unmaintained...
        const int thetaOffsetBase       = localIndexCalc * DIM + dimIndex;
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
    spread_charges<order, particlesPerBlock>(coefficient, kernelParams, globalIndex, localIndex,
                                             gridlineIndices, theta);
}

template <
    const int order
    >
__global__ void pme_wrap_kernel(const pme_gpu_kernel_params kernelParams)
{
    const int blockId = blockIdx.x
        + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
    const int            threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadLocalId;

    float * __restrict__ gridGlobal = kernelParams.grid.realGrid;

    const int            nx  = kernelParams.grid.localGridSize.x;
    const int            ny  = kernelParams.grid.localGridSize.y;
    const int            nz  = kernelParams.grid.localGridSize.z;
    const int            pny = kernelParams.grid.localGridSizePadded.y;
    const int            pnz = kernelParams.grid.localGridSizePadded.z;

    // should use ldg.128

    if (threadId < kernelParams.grid.overlapCellCounts[OVERLAP_ZONES - 1])
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
        const int ixy         = cellIndex / zoneSizeYZ.y; //yupinov check expensive integer divisions everywhere!
        const int iz          = cellIndex - zoneSizeYZ.y * ixy;
        const int ix          = ixy / zoneSizeYZ.x;
        const int iy          = ixy - zoneSizeYZ.x * ix;
        const int targetIndex = (ix * pny + iy) * pnz + iz;

        int       sourceOffset = 0;

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
        {
            atomicAdd(gridGlobal + targetIndex, gridGlobal[sourceIndex]);
        }
        else
        {
            gridGlobal[targetIndex] += gridGlobal[sourceIndex];
        }
    }
}

void pme_gpu_realloc_and_copy_fract_shifts(const gmx_pme_t *pme)
{
    cudaStream_t s = pme->gpu->pmeStream;

    const int    nx = pme->nkx; //replace
    const int    ny = pme->nky;
    const int    nz = pme->nkz;

    const int    cellCount = 5;
    /* This is the number of neighbor cells that is also hardcoded in make_gridindex5_to_localindex and should be the same */

    const int3 fshOffset = {0, cellCount * nx, cellCount * (nx + ny)};
    memcpy(&pme->gpu->kernelParams.grid.fshOffset, &fshOffset, sizeof(fshOffset));

    const int    newFractShiftsSize  = cellCount * (nx + ny + nz);

    /* Two arrays, same size */
    int currentSizeTemp      = pme->gpu->fractShiftsSize;
    int currentSizeTempAlloc = pme->gpu->fractShiftsSizeAlloc;
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.grid.fshArray, NULL, sizeof(float),
                        &currentSizeTemp, &currentSizeTempAlloc, newFractShiftsSize, pme->gpu->pmeStream, true);
    float *fshArray = pme->gpu->kernelParams.grid.fshArray;
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.grid.nnArray, NULL, sizeof(int),
                        &pme->gpu->fractShiftsSize, &pme->gpu->fractShiftsSizeAlloc, newFractShiftsSize, pme->gpu->pmeStream, true);
    int *nnArray = pme->gpu->kernelParams.grid.nnArray;

    //yupinov pinning?

    cu_copy_H2D_async(fshArray + fshOffset.x, pme->fshx, cellCount * nx * sizeof(float), s);
    cu_copy_H2D_async(fshArray + fshOffset.y, pme->fshy, cellCount * ny * sizeof(float), s);
    cu_copy_H2D_async(fshArray + fshOffset.z, pme->fshz, cellCount * nz * sizeof(float), s);

    cu_copy_H2D_async(nnArray + fshOffset.x, pme->nnx, cellCount * nx * sizeof(int), s);
    cu_copy_H2D_async(nnArray + fshOffset.y, pme->nny, cellCount * ny * sizeof(int), s);
    cu_copy_H2D_async(nnArray + fshOffset.z, pme->nnz, cellCount * nz * sizeof(int), s);

#if PME_USE_TEXTURES
    cudaError_t  stat;
#if PME_USE_TEXOBJ
    //if (use_texobj(dev_info))
    // should check device info here for CC >= 3.0
    {
        cudaResourceDesc rd;
        cudaTextureDesc  td;

        memset(&rd, 0, sizeof(rd));
        rd.resType                  = cudaResourceTypeLinear;
        rd.res.linear.devPtr        = fshArray;
        rd.res.linear.desc.f        = cudaChannelFormatKindFloat;
        rd.res.linear.desc.x        = 32;
        rd.res.linear.sizeInBytes   = newFractShiftsSize * sizeof(float);
        memset(&td, 0, sizeof(td));
        td.readMode                 = cudaReadModeElementType;
        stat = cudaCreateTextureObject(&pme->gpu->kernelParams.grid.fshTexture, &rd, &td, NULL);
        CU_RET_ERR(stat, "cudaCreateTextureObject on fsh_d failed");


        memset(&rd, 0, sizeof(rd));
        rd.resType                  = cudaResourceTypeLinear;
        rd.res.linear.devPtr        = nnArray;
        rd.res.linear.desc.f        = cudaChannelFormatKindSigned;
        rd.res.linear.desc.x        = 32;
        rd.res.linear.sizeInBytes   = newFractShiftsSize * sizeof(int);
        memset(&td, 0, sizeof(td));
        td.readMode                 = cudaReadModeElementType;
        stat = cudaCreateTextureObject(&pme->gpu->kernelParams.grid.nnTexture, &rd, &td, NULL);
        CU_RET_ERR(stat, "cudaCreateTextureObject on nn_d failed");
    }
    //else
#else
    {
        cudaChannelFormatDesc cd_fsh = cudaCreateChannelDesc<float>();
        stat = cudaBindTexture(NULL, &fshTextureRef, fshArray, &cd_fsh, newFractsShiftSize * sizeof(float)); //yupinov check overalloc
        CU_RET_ERR(stat, "cudaBindTexture on fsh failed");

        cudaChannelFormatDesc cd_nn = cudaCreateChannelDesc<int>();
        stat = cudaBindTexture(NULL, &nnTextureRef, nnArray, &cd_nn, newFractShiftsSize * sizeof(int));
        CU_RET_ERR(stat, "cudaBindTexture on nn failed");
        //yupinov unbind
    }
#endif
#endif
}

void pme_gpu_free_fract_shifts(const gmx_pme_t *pme)
{
    /* Two arrays, same size */
    cu_free_buffered(pme->gpu->kernelParams.grid.fshArray);
    cu_free_buffered(pme->gpu->kernelParams.grid.nnArray, &pme->gpu->fractShiftsSize, &pme->gpu->fractShiftsSizeAlloc);
    /* TODO: unbind textures here! */
}

void spread_on_grid_gpu(const gmx_pme_t *pme, pme_atomcomm_t gmx_unused *atc,
                        const int gmx_unused grid_index,
                        pmegrid_t *pmegrid,
                        const gmx_bool bCalcSplines,
                        const gmx_bool bSpread,
                        const gmx_bool bDoSplines)
{
    const gmx_bool bSeparateKernels = false;  // significantly slower if true
    if (!bCalcSplines && !bSpread)
    {
        gmx_fatal(FARGS, "No splining or spreading to be done?"); //yupinov use of gmx_fatal

    }
    //yupinov
    // bCalcSplines is always true - untested, unfinished
    // bDoSplines is always false - untested
    // bSpread is always true - untested, unfinished

    cudaStream_t s = pme->gpu->pmeStream;

    //int nx = pmegrid->s[XX], ny = pmegrid->s[YY], nz = pmegrid->s[ZZ];
    const int order   = pmegrid->order;
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
    const int nx  = pme->nkx;
    const int ny  = pme->nky;
    const int nz  = pme->nkz;

    const int n = pme->gpu->kernelParams.atoms.nAtoms;

    const int gridSize = pnx * pny * pnz * sizeof(float);

    // each spread kernel thread works on [order] contiguous x grid points, so we multiply the total number of threads by [order^2]
    // so only [1/order^2] of all kernel threads works on particle splines -> does it make sense to split it like this

    const int blockSize               = THREADS_PER_BLOCK;
    const int particlesPerBlock       = blockSize / order / order;
    const int splineParticlesPerBlock = particlesPerBlock; //blockSize / DIM; - can be easily changed, just have to pass spread theta stride to the spline kernel!
    // duplicated below!

    dim3 nBlocksSpread((n + blockSize - 1) / blockSize * order * order);
    dim3 nBlocksSpline((n + splineParticlesPerBlock - 1) / splineParticlesPerBlock);
    dim3 dimBlockSpread(order, order, particlesPerBlock); // used for spline_and_spread / spread
    dim3 dimBlockSpline(splineParticlesPerBlock, DIM);    // used for spline
    switch (order)
    {
        case 4:
            if (bSeparateKernels)
            {
                if (bCalcSplines)
                {
                    pme_gpu_timing_start(pme, gtPME_SPLINE);
                    if (bDoSplines)
                    {
                        gmx_fatal(FARGS, "the code for bDoSplines==true was not tested!");
                    }
                    else
                    {
                        pme_spline_kernel<4, blockSize / 4 / 4, FALSE> <<< nBlocksSpline, dimBlockSpline, 0, s>>> (pme->gpu->kernelParams);
                    }
                    CU_LAUNCH_ERR("pme_spline_kernel");
                    pme_gpu_timing_stop(pme, gtPME_SPLINE);
                }
                if (bSpread)
                {
                    pme_gpu_timing_start(pme, gtPME_SPREAD);
                    pme_spread_kernel<4, blockSize / 4 / 4> <<< nBlocksSpread, dimBlockSpread, 0, s>>> (pme->gpu->kernelParams);
                    CU_LAUNCH_ERR("pme_spread_kernel");
                    pme_gpu_timing_stop(pme, gtPME_SPREAD);
                }
            }
            else // a single monster kernel here
            {
                pme_gpu_timing_start(pme, gtPME_SPLINEANDSPREAD);
                if (bCalcSplines)
                {
                    if (bDoSplines)
                    {
                        gmx_fatal(FARGS, "the code for bDoSplines==true was not tested!");
                    }
                    else
                    {
                        if (bSpread)
                        {
                            pme_spline_and_spread_kernel<4, blockSize / 4 / 4, TRUE, FALSE, TRUE> <<< nBlocksSpread, dimBlockSpread, 0, s>>> (pme->gpu->kernelParams);
                        }
                        else
                        {
                            gmx_fatal(FARGS, "the code for bSpread==false was not tested!");
                        }
                    }
                }
                else
                {
                    gmx_fatal(FARGS, "the code for bCalcSplines==false was not tested!");
                }
                CU_LAUNCH_ERR("pme_spline_and_spread_kernel");
                pme_gpu_timing_stop(pme, gtPME_SPLINEANDSPREAD);
            }
            if (bSpread && pme->gpu->bGPUSingle)
            {
                // wrap on GPU as a separate small kernel - we need a complete grid first!
                const int blockSize       = 4 * warp_size; //yupinov this is everywhere! and architecture-specific
                const int overlappedCells = (nx + overlap) * (ny + overlap) * (nz + overlap) - nx * ny * nz;
                const int nBlocks         = (overlappedCells + blockSize - 1) / blockSize;

                pme_gpu_timing_start(pme, gtPME_WRAP);
                pme_wrap_kernel<4> <<< nBlocks, blockSize, 0, s>>> (pme->gpu->kernelParams);
                CU_LAUNCH_ERR("pme_wrap_kernel");
                pme_gpu_timing_stop(pme, gtPME_WRAP);
            }
            break;

        default:
            gmx_fatal(FARGS, "the code for pme_order != 4 was not tested!");
    }

    if (!pme->gpu->bGPUFFT && bSpread)
    {
        cu_copy_D2H_async(pmegrid->grid, pme->gpu->kernelParams.grid.realGrid, gridSize, s);
        cudaError_t stat = cudaEventRecord(pme->gpu->syncSpreadGridD2H, s);
        CU_RET_ERR(stat, "PME spread grid sync fail");
    }
    /*
       if (!pme->gpu->bGPUGather)
       {
        // FIXME: spline parameters layout is not the same on GPU => this would would fail with CPU gather.
        // Also no accounting for PME communication (bGPUSingle check?)
        for (int j = 0; j < DIM; ++j)
        {
            cu_copy_D2H_async(atc->spline[0].dtheta[j], dtheta_d + j * n * order, size_order, s);
            cu_copy_D2H_async(atc->spline[0].theta[j], theta_d + j * n * order, size_order, s);
        }
        cu_copy_D2H_async(atc->idx, gridlineIndicesGlobal, idx_size, s);
       }
     */
}
