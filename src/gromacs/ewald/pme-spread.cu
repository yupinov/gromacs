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
#include "pme-gpu.h"

#include <assert.h>

//yupinov optimizing unused parameters away?
//have to debug all boolean params

#define THREADS_PER_BLOCK   (4 * warp_size)
#define MIN_BLOCKS_PER_MP   (16)

#define USE_TEXTURES 1

//move all this into structure?
#if USE_TEXTURES
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
        const gmx_bool bCalcSplines, // first part
        const gmx_bool bDoSplines,   // bypassing conditional in the first part
        const gmx_bool bSpread       // second part
        >
//#if GMX_PTX_ARCH <= 300
__launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
//#endif
//yupinov put bounds on separate kernels as well
__global__ void pme_spline_and_spread_kernel
(const float3 nXYZ,
 int start_ix, int start_iy, int start_iz,
 const int pny, const int pnz,
 const int3 nnOffset,
#if USE_TEXTURES
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
#if !PME_EXTERN_CMEM
 const struct pme_gpu_recipbox_t RECIPBOX,
#endif
 int n)
{
/*

    pnx = pmegrid->s[XX];
    pny = pmegrid->s[YY];
    pnz = pmegrid->s[ZZ];

    offx = pmegrid->offset[XX];
    offy = pmegrid->offset[YY];
    offz = pmegrid->offset[ZZ];

*/

    const int offx = 0, offy = 0, offz = 0; //yupinov fix me!

    const int thetaStride = particlesPerBlock * DIM;

    __shared__ int idx[thetaStride];
    __shared__ real fractX[thetaStride];
    __shared__ real coefficient[particlesPerBlock];

    __shared__ real theta[thetaStride * order];
    __shared__ real dtheta[thetaStride * order];

    int ithx, index_x, ithy, ithz;

    const int localIndex = threadIdx.x;
    const int globalParticleIndexBase = blockIdx.x * particlesPerBlock;
    const int globalIndex = globalParticleIndexBase + localIndex;

    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;

    const int localIndexCalc = threadLocalId / DIM; // 4 instead of DIM
    const int dimIndex = threadLocalId - localIndexCalc * DIM;
    const int globalIndexCalc = globalParticleIndexBase + localIndexCalc;
    if (bCalcSplines)
    {
        // INTERPOLATION INDICES

        if ((globalIndexCalc < n) && (localIndexCalc < particlesPerBlock) && (dimIndex < DIM))
        {
            int constIndex, tInt;
            real n, t;
            const float3 x = coordinatesGlobal[globalIndexCalc];
            // accessing fields in nnOffset/nXYZ/RECIPBOX/... with dimIndex offset
            // puts them into local memory (!) instead of accessing the constant memory directly
            // that's the reason for the switch
            switch (dimIndex)
            {
                case 0:
                constIndex = nnOffset.x;
                n = nXYZ.x;
                t = x.x * RECIPBOX.box[dimIndex].x + x.y * RECIPBOX.box[dimIndex].y + x.z * RECIPBOX.box[dimIndex].z;
                break;

                case 1:
                constIndex = nnOffset.y;
                n = nXYZ.y;
                t = /*x.x * RECIPBOX.box[dimIndex].x + */ x.y * RECIPBOX.box[dimIndex].y + x.z * RECIPBOX.box[dimIndex].z;
                break;

                case 2:
                constIndex = nnOffset.z;
                n = nXYZ.z;
                t = /*x.x * RECIPBOX.box[dimIndex].x + x.y * RECIPBOX.box[dimIndex].y + */ x.z * RECIPBOX.box[dimIndex].z;
                break;
            }
            // parts of multiplication are commented because these components are actually 0
            // thus, excessive constant memory
            // should refactor if settling for this approach...

            // Fractional coordinates along box vectors, add 2.0 to make 100% sure we are positive for triclinic boxes
            t = (t + 2.0f) * n;
            tInt = (int)t;
            fractX[threadLocalId] = t - tInt;
            constIndex += tInt;

#if USE_TEXTURES
#if USE_TEXOBJ
            fractX[threadLocalId] += tex1Dfetch<real>(fshTexture, constIndex);
            idx[threadLocalId] = tex1Dfetch<int>(nnTexture, constIndex);
#else
            fractX[threadLocalId] += tex1Dfetch(fshTextureRef, constIndex);
            idx[threadLocalId] = tex1Dfetch(nnTextureRef, constIndex);
#endif
#else
            fractX[threadLocalId] += fsh[constIndex];
            idx[threadLocalId] = nn[constIndex];
#endif
            // staging for both parts

            idxGlobal[globalIndexCalc * DIM + dimIndex] = idx[threadLocalId]; //yupinov fix indexing
            if (threadLocalId < particlesPerBlock)
                coefficient[threadLocalId] = coefficientGlobal[globalParticleIndexBase + threadLocalId];
        }
        __syncthreads();

        // MAKE BSPLINES
        if ((globalIndexCalc < n) && (localIndexCalc < particlesPerBlock) && (dimIndex < DIM)) // just for sync?
        {
            if (bDoSplines || (coefficient[localIndex] != 0.0f)) //yupinov how bad is this conditional?
            {
                real dr, div;
                real data[order];

                dr = fractX[threadLocalId];

                /* dr is relative offset from lower cell limit */
                data[order - 1] = 0;
                data[1]         = dr;
                data[0]         = 1 - dr;

#pragma unroll
                for (int k = 3; k < order; k++)
                {
                    div         = 1.0f / (k - 1.0f);
                    data[k - 1] = div * dr * data[k - 2];
#pragma unroll
                    for (int l = 1; l < (k - 1); l++)
                    {
                        data[k - l - 1] = div * ((dr + l) * data[k - l - 2] + (k - l - dr) * data[k - l - 1]);
                    }
                    data[0] = div * (1 - dr) * data[0];
                }
                /* differentiate */
                const int thetaOffsetBase = localIndexCalc * DIM + dimIndex;
                dtheta[thetaOffsetBase] = -data[0];

#pragma unroll
                for (int k = 1; k < order; k++)
                {
                    dtheta[thetaOffsetBase + k * thetaStride] = data[k - 1] - data[k];
                }

                div             = 1.0f / (order - 1);
                data[order - 1] = div * dr * data[order - 2];
#pragma unroll
                for (int l = 1; l < (order - 1); l++)
                {
                    data[order - l - 1] = div * ((dr + l) * data[order - l - 2] + (order - l - dr) * data[order - l - 1]);
                }
                data[0] = div * (1 - dr) * data[0];

#pragma unroll
                for (int k = 0; k < order; k++)
                {
                    theta[thetaOffsetBase + k * thetaStride] = data[k];
                }

                //yupinov store to global
                const int thetaGlobalOffsetBase = globalParticleIndexBase * DIM * order;
#pragma unroll
                for (int k = 0; k < order; k++)
                {
                    const int thetaIndex = thetaOffsetBase + k * thetaStride;

                    thetaGlobal[thetaGlobalOffsetBase + thetaIndex] = theta[thetaIndex];
                    dthetaGlobal[thetaGlobalOffsetBase + thetaIndex] = dtheta[thetaIndex];
                }
            }
        }
        __syncthreads(); //yupinov do we need it?
    }
    else if (bSpread) // staging for spread
    {
        if ((globalIndexCalc < n) && (dimIndex < DIM) && (localIndexCalc < particlesPerBlock))
        {
            idx[localIndexCalc * DIM + dimIndex] = idxGlobal[globalIndexCalc * DIM + dimIndex]; //yupinov check instructions

            const int thetaOffsetBase = localIndexCalc * DIM + dimIndex;
            const int thetaGlobalOffsetBase = globalParticleIndexBase * DIM * order;
#pragma unroll
            for (int k = 0; k < order; k++)
            {
                const int thetaIndex = thetaOffsetBase + k * thetaStride;
                theta[thetaIndex] = thetaGlobal[thetaGlobalOffsetBase + thetaIndex];
            }

            if (threadLocalId < particlesPerBlock)
                coefficient[threadLocalId] = coefficientGlobal[globalParticleIndexBase + threadLocalId];
        }
        __syncthreads();
    }

    // SPREAD

    // spline Y/Z coordinates
    ithy = threadIdx.y;
    ithz = threadIdx.z;

    if (bSpread)
    {
        if ((globalIndex < n) && (coefficient[localIndex] != 0.0f)) //yupinov store checks
        {
            const int ix = idx[localIndex * DIM + XX] - offx;
            const int iy = idx[localIndex * DIM + YY] - offy;
            const int iz = idx[localIndex * DIM + ZZ] - offz;

            const int thetaOffsetBase = localIndex * DIM;
            const real thz = theta[thetaOffsetBase + ithz * thetaStride + ZZ];
            const real thy = theta[thetaOffsetBase + ithy * thetaStride + YY];
            const real constVal = thz * thy * coefficient[localIndex];
            const int constOffset = (iy + ithy) * pnz + (iz + ithz);
            const real *thx = theta + (thetaOffsetBase + XX);

#pragma unroll
            for (ithx = 0; (ithx < order); ithx++)
            {
                index_x = (ix + ithx) * pny * pnz;
                atomicAdd(gridGlobal + index_x + constOffset, thx[ithx * thetaStride] * constVal);
            }
        }
    }
}


// pme_spline_and_spread split into pme_spline and pme_spread - as an experiment

template <
        const int order,
        const int particlesPerBlock,
        const gmx_bool bDoSplines
        >
__global__ void pme_spline_kernel
(const float3 nXYZ,
 const int start_ix, const int start_iy, const int start_iz,
 const int3 nnOffset,
#if USE_TEXTURES
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
 #if !PME_EXTERN_CMEM
  const struct pme_gpu_recipbox_t RECIPBOX,
 #endif
 const int n)
{
/*

    pnx = pmegrid->s[XX];
    pny = pmegrid->s[YY];
    pnz = pmegrid->s[ZZ];

    offx = pmegrid->offset[XX];
    offy = pmegrid->offset[YY];
    offz = pmegrid->offset[ZZ];

*/

    const int thetaStride = particlesPerBlock * DIM;

    __shared__ int idx[thetaStride];
    __shared__ real fractX[thetaStride];
    __shared__ real coefficient[particlesPerBlock];

    __shared__ real theta[thetaStride * order];
    __shared__ real dtheta[thetaStride * order];

    const int localIndex = threadIdx.x;
    const int globalParticleIndexBase = blockIdx.x * particlesPerBlock;

    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;

    const int localIndexCalc = threadLocalId / DIM; // 4 instead of DIM
    const int dimIndex = threadLocalId - localIndexCalc * DIM;

    const int globalIndexCalc = globalParticleIndexBase + localIndexCalc;

    // INTERPOLATION INDICES
    if ((globalIndexCalc < n) && (localIndexCalc < particlesPerBlock) && (dimIndex < DIM))
    //yupinov - this is a single particle work!
    {
        int constIndex, tInt;
        real n, t;
        const float3 x = coordinatesGlobal[globalIndexCalc];
        // accessing fields in nnOffset/nXYZ/RECIPBOX/... with dimIndex offset
        // puts them into local memory (!) instead of accessing the constant memory directly
        // that's the reason for the switch
        switch (dimIndex)
        {
            case 0:
            constIndex = nnOffset.x;
            n = nXYZ.x;
            t = x.x * RECIPBOX.box[dimIndex].x + x.y * RECIPBOX.box[dimIndex].y + x.z * RECIPBOX.box[dimIndex].z;
            break;

            case 1:
            constIndex = nnOffset.y;
            n = nXYZ.y;
            t = /*x.x * RECIPBOX.box[dimIndex].x + */ x.y * RECIPBOX.box[dimIndex].y + x.z * RECIPBOX.box[dimIndex].z;
            break;

            case 2:
            constIndex = nnOffset.z;
            n = nXYZ.z;
            t = /*x.x * RECIPBOX.box[dimIndex].x + x.y * RECIPBOX.box[dimIndex].y + */ x.z * RECIPBOX.box[dimIndex].z;
            break;
        }
        // parts of multiplication are commented because these components are actually 0
        // thus, excessive constant memory
        // should refactor if settling for this approach...

        // Fractional coordinates along box vectors, add 2.0 to make 100% sure we are positive for triclinic boxes
        t = (t + 2.0f) * n;
        tInt = (int)t;
        fractX[threadLocalId] = t - tInt;
        constIndex += tInt;

#if USE_TEXTURES
#if USE_TEXOBJ
        fractX[threadLocalId] += tex1Dfetch<real>(fshTexture, constIndex);
        idx[threadLocalId] = tex1Dfetch<int>(nnTexture, constIndex);
#else
        fractX[threadLocalId] += tex1Dfetch(fshTextureRef, constIndex);
        idx[threadLocalId] = tex1Dfetch(nnTextureRef, constIndex);
#endif
#else
        fractX[threadLocalId] += fsh[constIndex];
        idx[threadLocalId] = nn[constIndex];
#endif

        //staging for both parts

         idxGlobal[globalIndexCalc * DIM + dimIndex] = idx[threadLocalId]; //yupinov fix indexing
         if (threadLocalId < particlesPerBlock)
             coefficient[threadLocalId] = coefficientGlobal[globalParticleIndexBase + threadLocalId];
    }
    __syncthreads();

    // MAKE BSPLINES
    if ((globalIndexCalc < n) && (localIndexCalc < particlesPerBlock) && (dimIndex < DIM)) // just for sync?
    {
        if (bDoSplines || (coefficient[localIndex] != 0.0f)) //yupinov how bad is this conditional?
        {
            real dr, div;
            real data[order];

            dr = fractX[threadLocalId];

            /* dr is relative offset from lower cell limit */
            data[order - 1] = 0;
            data[1]         = dr;
            data[0]         = 1 - dr;

#pragma unroll
            for (int k = 3; k < order; k++)
            {
                div         = 1.0f / (k - 1.0f);
                data[k - 1] = div * dr * data[k - 2];
                #pragma unroll
                for (int l = 1; l < (k - 1); l++)
                {
                    data[k - l - 1] = div * ((dr + l) * data[k - l - 2] + (k - l - dr) * data[k - l - 1]);
                }
                data[0] = div * (1 - dr) * data[0];
            }
            /* differentiate */
            const int thetaOffsetBase = localIndexCalc * DIM + dimIndex;
            dtheta[thetaOffsetBase] = -data[0];

#pragma unroll
            for (int k = 1; k < order; k++)
            {
                dtheta[thetaOffsetBase + k * thetaStride] = data[k - 1] - data[k];
            }

            div             = 1.0f / (order - 1);
            data[order - 1] = div * dr * data[order - 2];
#pragma unroll
            for (int l = 1; l < (order - 1); l++)
            {
                data[order - l - 1] = div * ((dr + l) * data[order - l - 2] + (order - l - dr) * data[order - l - 1]);
            }
            data[0] = div * (1 - dr) * data[0];

#pragma unroll
            for (int k = 0; k < order; k++)
            {
                theta[thetaOffsetBase + k * thetaStride] = data[k];
            }

            //yupinov store to global
            const int thetaGlobalOffsetBase = globalParticleIndexBase * DIM * order;
#pragma unroll
            for (int k = 0; k < order; k++)
            {
                const int thetaIndex = thetaOffsetBase + k * thetaStride;

                thetaGlobal[thetaGlobalOffsetBase + thetaIndex] = theta[thetaIndex];
                dthetaGlobal[thetaGlobalOffsetBase + thetaIndex] = dtheta[thetaIndex];
            }
        }
    }
}


template <const int order, const int particlesPerBlock>
__global__ void pme_spread_kernel
(int start_ix, int start_iy, int start_iz,
  const int pny, const int pnz,
 const real * __restrict__ coefficientGlobal,
 real * __restrict__ gridGlobal, real * __restrict__ thetaGlobal, const int * __restrict__ idxGlobal,
 int n)
{
/*

    pnx = pmegrid->s[XX];
    pny = pmegrid->s[YY];
    pnz = pmegrid->s[ZZ];

    offx = pmegrid->offset[XX];
    offy = pmegrid->offset[YY];
    offz = pmegrid->offset[ZZ];

*/
    const int thetaStride = particlesPerBlock * DIM;

    const int offx = 0, offy = 0, offz = 0;//yupinov fix me!

    __shared__ int idx[thetaStride];
    __shared__ real coefficient[particlesPerBlock];

    __shared__ real theta[thetaStride * order];

    int ithx, index_x, ithy, ithz;

    const int localIndex = threadIdx.x;
    const int globalParticleIndexBase = blockIdx.x * particlesPerBlock;
    const int globalIndex = globalParticleIndexBase + localIndex;


    // staging
    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;
    const int localIndexCalc = threadLocalId / DIM;
    const int dimIndex = threadLocalId - localIndexCalc * DIM;
    const int globalIndexCalc = globalParticleIndexBase + localIndexCalc;
    if ((globalIndexCalc < n) && (dimIndex < DIM) && (localIndexCalc < particlesPerBlock))
    {
        idx[localIndexCalc * DIM + dimIndex] = idxGlobal[globalIndexCalc * DIM + dimIndex]; //yupinov check instructions

        const int thetaOffsetBase = localIndexCalc * DIM + dimIndex;
        const int thetaGlobalOffsetBase = globalParticleIndexBase * DIM * order;
#pragma unroll
        for (int k = 0; k < order; k++)
        {
            const int thetaIndex = thetaOffsetBase + k * thetaStride;
            theta[thetaIndex] = thetaGlobal[thetaGlobalOffsetBase + thetaIndex];
        }

        if (threadLocalId < particlesPerBlock)
            coefficient[threadLocalId] = coefficientGlobal[globalParticleIndexBase + threadLocalId];
    }
    __syncthreads();

    // SPREAD

    // spline Y/Z coordinates
    ithy = threadIdx.y;
    ithz = threadIdx.z;

    if ((globalIndex < n) && (coefficient[localIndex] != 0.0f)) //yupinov store checks
    {
        const int ix = idx[localIndex * DIM + XX] - offx;
        const int iy = idx[localIndex * DIM + YY] - offy;
        const int iz = idx[localIndex * DIM + ZZ] - offz;

        const int thetaOffsetBase = localIndex * DIM;
        const real thz = theta[thetaOffsetBase + ithz * thetaStride + ZZ];
        const real thy = theta[thetaOffsetBase + ithy * thetaStride + YY];
        const real constVal = thz * thy * coefficient[localIndex];
        const int constOffset = (iy + ithy) * pnz + (iz + ithz);
        const real *thx = theta + (thetaOffsetBase + XX);

#pragma unroll
        for (ithx = 0; (ithx < order); ithx++)
        {
            index_x = (ix + ithx) * pny * pnz;
            atomicAdd(gridGlobal + index_x + constOffset, thx[ithx * thetaStride] * constVal);
        }
    }
}

template <
    const int order
    >
__global__ void pme_wrap_kernel
    (const int nx, const int ny, const int nz,
     const int pny, const int pnz,
#if !PME_EXTERN_CMEM
    const pme_gpu_overlap_t OVERLAP,
#endif
     real * __restrict__ grid
     )
{
    // const int overlap = order - 1;

    // WRAP
    const int blockId = blockIdx.x
                 + blockIdx.y * gridDim.x
                 + gridDim.x * gridDim.y * blockIdx.z;
    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;
    const int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadLocalId;

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
    real *fshArray = pme->gpu->fshArray = (real *)PMEMemoryFetch(PME_ID_FSH, fshSize, ML_DEVICE);
    cu_copy_H2D_async(fshArray                , pme->fshx, 5 * nx * sizeof(real), s);
    cu_copy_H2D_async(fshArray + 5 * nx       , pme->fshy, 5 * ny * sizeof(real), s);
    cu_copy_H2D_async(fshArray + 5 * (nx + ny), pme->fshz, 5 * nz * sizeof(real), s);

    const int nnSize = 5 * (nx + ny + nz) * sizeof(int);
    int *nnArray = pme->gpu->nnArray = (int *)PMEMemoryFetch(PME_ID_NN, nnSize, ML_DEVICE);
    cu_copy_H2D_async(nnArray                , pme->nnx, 5 * nx * sizeof(int), s);
    cu_copy_H2D_async(nnArray + 5 * nx       , pme->nny, 5 * ny * sizeof(int), s);
    cu_copy_H2D_async(nnArray + 5 * (nx + ny), pme->nnz, 5 * nz * sizeof(int), s);

#if USE_TEXTURES
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

void pme_gpu_alloc_grid(struct gmx_pme_t *pme, const int grid_index)
{
    const int pnx = pme->pmegrid_nx;
    const int pny = pme->pmegrid_ny;
    const int pnz = pme->pmegrid_nz;
    const int gridSize = pnx * pny * pnz * sizeof(real);

    pme->gpu->grid = (real *)PMEMemoryFetch(PME_ID_REAL_GRID, gridSize, ML_DEVICE);
}

void pme_gpu_clear_grid(struct gmx_pme_t *pme, const int grid_index)
{
    /*
    pmegrid_t *pmegrid = &(pme->pmegrid[grid_index].grid); //yupinov most PME GPU functions ignore grid indices anyway
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

void spread_on_grid_gpu(struct gmx_pme_t *pme, pme_atomcomm_t *atc,
         int grid_index,
         pmegrid_t *pmegrid,
         const gmx_bool bCalcSplines,
         const gmx_bool bSpread,
         const gmx_bool bDoSplines)
//yupinov templating!
//real *fftgrid
//added:, gmx_wallcycle_t wcycle)
{
    const gmx_bool separateKernels = false;  // significantly slower if true
    if (!bCalcSplines && !bSpread)
        gmx_fatal(FARGS, "No splining or spreading to be done?"); //yupinov use of gmx_fatal

    //yupinov
    // bCalcSplines is always true - untested, unfinished
    // bDoSplines is always false - untested
    // bSpread is always true - untested, unfinished

    cudaStream_t s = pme->gpu->pmeStream;

    atc->spline[0].n = atc->n; //yupinov - without it, the conserved energy went down by 0.5%! used in gather or sometwhere else?

    //int nx = pmegrid->s[XX], ny = pmegrid->s[YY], nz = pmegrid->s[ZZ];
    const int order = pmegrid->order;
    const int overlap = order - 1;

    /*
    ivec local_ndata, local_size, local_offset;
    gmx_parallel_3dfft_real_limits_wrapper(pme, grid_index, local_ndata, local_offset, local_size);
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

    int n = atc->n;
    const int gridSize = pnx * pny * pnz * sizeof(real);

    int size_order = order * n * sizeof(real);
    int size_order_dim = size_order * DIM;
    real *theta_d = (real *)PMEMemoryFetch(PME_ID_THETA, size_order_dim, ML_DEVICE);
    real *dtheta_d = (real *)PMEMemoryFetch(PME_ID_DTHETA, size_order_dim, ML_DEVICE);

    // IDXPTR
    int idx_size = n * DIM * sizeof(int);
    int *idx_d = (int *)PMEMemoryFetch(PME_ID_IDXPTR, idx_size, ML_DEVICE);

    const float3 nXYZ = {(real)nx, (real)ny, (real)nz};
    const int3 nnOffset = {0, 5 * nx, 5 * (nx + ny)};


    if (bCalcSplines)
    {
         /*
        const size_t coordinatesSize = DIM * n_blocked * sizeof(real);
        float3 *xptr_h = (float3 *)PMEMemoryFetch(PME_ID_XPTR, coordinatesSize, ML_HOST);
        memcpy(xptr_h, atc->x, coordinatesSize);
        xptr_d = (float3 *)PMEMemoryFetch(PME_ID_XPTR, coordinatesSize, ML_DEVICE);
        cu_copy_H2D_async(xptr_d, xptr_h, coordinatesSize, ML_DEVICE, pme->gpu->pmeStream);
        */
    }


    //yupinov blocked approach everywhere or nowhere
    //filtering?
    /*

    const size_t coefficientSize = n * sizeof(real);
    real *coefficient_h = (real *)PMEMemoryFetch(PME_ID_COEFFICIENT, coefficientSize, ML_HOST);
    memcpy(coefficient_h, atc->coefficient, coefficientSize);
    real *coefficient_d = (real *)PMEMemoryFetch(PME_ID_COEFFICIENT, coefficientSize, ML_DEVICE);
    cu_copy_H2D_async(coefficient_d, coefficient_h, coefficientSize, s);
    */

    // in spread-unified each kernel thread works on one particle: calculates its splines, spreads it to [order^3] gridpoints
    // here each kernel thread works on [order] contiguous x grid points, so we multiply the total number of threads by [order^2]
    // so only [1/order^2] of all kernel threads works on particle splines -> does it make sense to split it like this


    //const int particlesPerBlock = warp_size;
    const int blockSize = THREADS_PER_BLOCK;
    const int particlesPerBlock = blockSize / order / order;
    //duplicated below!

    //this is the number of particles for SPREAD, btw
    dim3 nBlocks((n + blockSize - 1) / blockSize * order * order, 1, 1);
    //dim3 dimBlock(order, order, D); //each block has 32 threads now to hand 32 particlesPerBlock
    //dim3 dimBlock(particlesPerBlock, 1, 1);
    dim3 dimBlock(particlesPerBlock, order, order);
    switch (order)
    {
        case 4:
            if (separateKernels)
            {
                if (bCalcSplines)
                {
                    pme_gpu_timing_start(pme, ewcsPME_SPLINE);

                    if (bDoSplines)
                        gmx_fatal(FARGS, "the code for bDoSplines==true was not tested!");
                    else
                    {
                        pme_spline_kernel<4, blockSize / 4 / 4, FALSE> <<<nBlocks, dimBlock, 0, s>>>
                                                                                                   (nXYZ,
                                                                                                    pme->pmegrid_start_ix, pme->pmegrid_start_iy, pme->pmegrid_start_iz,
                                                                                                    nnOffset,
#if USE_TEXTURES
#if USE_TEXOBJ
                                                                                                    nnTexture, fshTexture,
#endif
#else
                                                                                                    pme->gpu->nnArray, pme->gpu->fshArray,
#endif
                                                                                                    pme->gpu->coordinates,
                                                                                                    pme->gpu->coefficients,
                                                                                                    theta_d, dtheta_d, idx_d,
#if !PME_EXTERN_CMEM
                                                                                                    pme->gpu->recipbox,
#endif
                                                                                                    n);


                    }

                    CU_LAUNCH_ERR("pme_spline_kernel");

                    pme_gpu_timing_stop(pme, ewcsPME_SPLINE);
                }
                if (bSpread)
                {
                    pme_gpu_timing_start(pme, ewcsPME_SPREAD);

                    pme_spread_kernel<4, blockSize / 4 / 4> <<<nBlocks, dimBlock, 0, s>>>
                                                                            (pme->pmegrid_start_ix, pme->pmegrid_start_iy, pme->pmegrid_start_iz,
                                                                             pny, pnz,
                                                                             pme->gpu->coefficients,
                                                                             pme->gpu->grid, theta_d, idx_d,
                                                                             n);

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
                            pme_spline_and_spread_kernel<4, blockSize / 4 / 4, TRUE, FALSE, TRUE> <<<nBlocks, dimBlock, 0, s>>>
                                  (nXYZ,
                                   pme->pmegrid_start_ix, pme->pmegrid_start_iy, pme->pmegrid_start_iz,
                                   pny, pnz,
                                   nnOffset,
#if USE_TEXTURES
#if USE_TEXOBJ
                                   nnTexture, fshTexture,
#endif
#else
                                   pme->gpu->nnArray, pme->gpu->fshArray,
#endif
                                   pme->gpu->coordinates, pme->gpu->coefficients, pme->gpu->grid, theta_d, dtheta_d, idx_d,
#if !PME_EXTERN_CMEM
                                   pme->gpu->recipbox,
#endif
                                   n);
                        }
                        else
                            gmx_fatal(FARGS, "the code for bSpread==false was not tested!");
                    }
                }
                else
                    gmx_fatal(FARGS, "the code for bCalcSplines==false was not tested!"); //yupinov
                CU_LAUNCH_ERR("pme_spline_and_spread_kernel");

                pme_gpu_timing_stop(pme, ewcsPME_SPLINEANDSPREAD);
            }
            if (bSpread && pme->bGPUSingle)
            {
                // wrap on GPU as a separate small kernel - we need a complete grid first!
                const int blockSize = 4 * warp_size; //yupinov this is everywhere! and arichitecture-specific
                const int overlappedCells = (nx + overlap) * (ny + overlap) * (nz + overlap) - nx * ny * nz;
                const int nBlocks = (overlappedCells + blockSize - 1) / blockSize;

                pme_gpu_timing_start(pme, ewcsPME_WRAP);

                pme_wrap_kernel<4> <<<nBlocks, blockSize, 0, s>>>(nx, ny, nz, pny, pnz,
#if !PME_EXTERN_CMEM
                                                                  pme->gpu->overlap,
#endif
                                                                  pme->gpu->grid);

                CU_LAUNCH_ERR("pme_wrap_kernel");

                pme_gpu_timing_stop(pme, ewcsPME_WRAP);
            }
            break;

        default:
            gmx_fatal(FARGS, "the code for pme_order != 4 was not tested!"); //yupinov
    }

    if (!pme->gpu->keepGPUDataBetweenSpreadAndR2C)
    {
        if (bSpread)
            cu_copy_D2H_async(pmegrid->grid, pme->gpu->grid, gridSize, s);
        for (int j = 0; j < DIM; ++j) //also breaking compacting in gather
        //and why not just check bGPUSingle here?
        {
            cu_copy_D2H_async(atc->spline[0].dtheta[j], dtheta_d + j * n * order, size_order, s);
            cu_copy_D2H_async(atc->spline[0].theta[j], theta_d + j * n * order, size_order, s);
        }
        //yupinov what about theta/dtheta/idx use in pme_realloc_atomcomm_things?
        cu_copy_D2H_async(atc->idx, idx_d, idx_size, s);
    }
    //yupinov check flags like bSpread etc. before copying...
}

