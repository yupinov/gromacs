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

gpu_events gpu_events_spline;
gpu_events gpu_events_spread;
gpu_events gpu_events_splineandspread;

//yupinov optimizing unused parameters away?
//have to debug all boolean params

#define THREADS_PER_BLOCK   (4 * warp_size)
#define MIN_BLOCKS_PER_MP   (16)

#define USE_TEXTURES 1
//textures seems just a bit slower on GTX 660 Ti, so I'm keeping this define just in case

#if USE_TEXTURES
#define USE_TEXOBJ 0
#if USE_TEXOBJ
cudaTextureObject_t nnTexture;
cudaTextureObject_t fshTexture;
#else
texture<int, 1, cudaReadModeElementType> nnTextureRef;
texture<float, 1, cudaReadModeElementType> fshTextureRef;
#endif
#endif
//yupinov

__constant__ __device__ float3 RECIPBOX[3];

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
 const float3 * __restrict__ xptr,
 const real * __restrict__ coefficientGlobal,
 real * __restrict__ grid, real * __restrict__ theta, real * __restrict__ dtheta, int * __restrict__ idx, //yupinov
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

    __shared__ int idxShared[thetaStride];
    __shared__ real fractX[thetaStride];
    __shared__ real coefficient[particlesPerBlock];

    __shared__ real theta_shared[thetaStride * order];
    __shared__ real dtheta_shared[thetaStride * order];

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

        __shared__ real t[thetaStride];
        __shared__ int tInt[thetaStride];
        if ((globalIndexCalc < n) && (localIndexCalc < particlesPerBlock) && (dimIndex < DIM))
        {
            int constIndex;
            real n;
            // we're doing this switch because accesing field in nnOffset/nXYZ directly with dimIndex offset puts them into registers instead of accesing the constant memory directly
            switch (dimIndex)
            {
                case 0:
                constIndex = nnOffset.x;
                n = nXYZ.x;
                break;

                case 1:
                constIndex = nnOffset.y;
                n = nXYZ.y;
                break;

                case 2:
                constIndex = nnOffset.z;
                n = nXYZ.z;
                break;
            }

            const float3 x = xptr[globalIndexCalc];
            const float3 recip = RECIPBOX[dimIndex];
            // Fractional coordinates along box vectors, add 2.0 to make 100% sure we are positive for triclinic boxes
            t[threadLocalId] = (x.x * recip.x + x.y * recip.y + x.z * recip.z + 2.0f) * n;
            tInt[threadLocalId] = (int)t[threadLocalId]; //yupinov test registers
            fractX[threadLocalId] = t[threadLocalId] - tInt[threadLocalId];

            /* Because decomposition only occurs in x and y,
            * we never have a fraction correction in z.
            */

            constIndex += tInt[threadLocalId];
#if USE_TEXTURES
#if USE_TEXOBJ
            fractX[threadLocalId] += tex1Dfetch<real>(fshTexture, constIndex);
            idxShared[threadLocalId] = tex1Dfetch<int>(nnTexture, constIndex);
#else
            fractX[threadLocalId] += tex1Dfetch(fshTextureRef, constIndex);
            idxShared[threadLocalId] = tex1Dfetch(nnTextureRef, constIndex);
#endif
#else
            fractX[threadLocalId] += fsh[constIndex];
            idxShared[threadLocalId] = nn[constIndex];
#endif
           //staging for both parts

            idx[globalIndexCalc * DIM + dimIndex] = idxShared[threadLocalId]; //yupinov fix indexing
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
                dtheta_shared[thetaOffsetBase] = -data[0];

#pragma unroll
                for (int k = 1; k < order; k++)
                {
                    dtheta_shared[thetaOffsetBase + k * thetaStride] = data[k - 1] - data[k];
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
                    theta_shared[thetaOffsetBase + k * thetaStride] = data[k];
                }

                //yupinov store to global
                const int thetaGlobalOffsetBase = globalParticleIndexBase * DIM * order;
#pragma unroll
                for (int k = 0; k < order; k++)
                {
                    const int thetaIndex = thetaOffsetBase + k * thetaStride;

                    theta[thetaGlobalOffsetBase + thetaIndex] = theta_shared[thetaIndex];
                    dtheta[thetaGlobalOffsetBase + thetaIndex] = dtheta_shared[thetaIndex];
                }
            }
        }
        __syncthreads(); //yupinov do we need it?
    }
    else if (bSpread) // staging for spread
    {
        if ((globalIndexCalc < n) && (dimIndex < DIM) && (localIndexCalc < particlesPerBlock))
        {
            idxShared[localIndexCalc * DIM + dimIndex] = idx[globalIndexCalc * DIM + dimIndex]; //yupinov check instructions

            const int thetaOffsetBase = localIndexCalc * DIM + dimIndex;
            const int thetaGlobalOffsetBase = globalParticleIndexBase * DIM * order;
#pragma unroll
            for (int k = 0; k < order; k++)
            {
                const int thetaIndex = thetaOffsetBase + k * thetaStride;
                theta_shared[thetaIndex] = theta[thetaGlobalOffsetBase + thetaIndex];
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
            const int ix = idxShared[localIndex * DIM + XX] - offx;
            const int iy = idxShared[localIndex * DIM + YY] - offy;
            const int iz = idxShared[localIndex * DIM + ZZ] - offz;

            const int thetaOffsetBase = localIndex * DIM;
            const real thz = theta_shared[thetaOffsetBase + ithz * thetaStride + ZZ];
            const real thy = theta_shared[thetaOffsetBase + ithy * thetaStride + YY];
            const real constVal = thz * thy * coefficient[localIndex];
            const int constOffset = (iy + ithy) * pnz + (iz + ithz);
            const real *thx = theta_shared + (thetaOffsetBase + XX);

#pragma unroll
            for (ithx = 0; (ithx < order); ithx++)
            {
                index_x = (ix + ithx) * pny * pnz;
                atomicAdd(grid + index_x + constOffset, thx[ithx * thetaStride] * constVal);
            }
        }
    }
}


// spline_and_spread split into spline and spread - as an experiment

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
 const float3 * __restrict__ xptr,
 const real * __restrict__ coefficientGlobal,
 real * __restrict__ theta, real * __restrict__ dtheta, int * __restrict__ idx, //yupinov
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

    __shared__ int idxShared[thetaStride];
    __shared__ real fractX[thetaStride];
    __shared__ real coefficient[particlesPerBlock];

    __shared__ real theta_shared[thetaStride * order];
    __shared__ real dtheta_shared[thetaStride * order];

    const int localIndex = threadIdx.x;
    const int globalParticleIndexBase = blockIdx.x * particlesPerBlock;

    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;

    const int localIndexCalc = threadLocalId / DIM; // 4 instead of DIM
    const int dimIndex = threadLocalId - localIndexCalc * DIM;

    const int globalIndexCalc = globalParticleIndexBase + localIndexCalc;
    __shared__ real t[thetaStride];
    __shared__ int tInt[thetaStride];

    // INTERPOLATION INDICES
    if ((globalIndexCalc < n) && (localIndexCalc < particlesPerBlock) && (dimIndex < DIM))
    //yupinov - this is a single particle work!
    {
        int constIndex;
        real n;
        // we're doing this switch because accesing fielsd in nnOffset/nXYZ directly with dimIndex offset puts them into registers instead of accessing the constant memory directly
        switch (dimIndex)
        {
            case 0:
            constIndex = nnOffset.x;
            n = nXYZ.x;
            break;

            case 1:
            constIndex = nnOffset.y;
            n = nXYZ.y;
            break;

            case 2:
            constIndex = nnOffset.z;
            n = nXYZ.z;
            break;
        }

        const float3 x = xptr[globalIndexCalc];
        const float3 recip = RECIPBOX[dimIndex];
        // Fractional coordinates along box vectors, add 2.0 to make 100% sure we are positive for triclinic boxes
        t[threadLocalId] = (x.x * recip.x + x.y * recip.y + x.z * recip.z + 2.0f) * n;
        tInt[threadLocalId] = (int)t[threadLocalId]; //yupinov test registers
        fractX[threadLocalId] = t[threadLocalId] - tInt[threadLocalId];

        constIndex += tInt[threadLocalId];
#if USE_TEXTURES
#if USE_TEXOBJ
            fractX[threadLocalId] += tex1Dfetch<real>(fshTexture, constIndex);
            idxShared[threadLocalId] = tex1Dfetch<int>(nnTexture, constIndex);
#else
            fractX[threadLocalId] += tex1Dfetch(fshTextureRef, constIndex);
            idxShared[threadLocalId] = tex1Dfetch(nnTextureRef, constIndex);
#endif
#else
            fractX[threadLocalId] += fsh[constIndex];
            idxShared[threadLocalId] = nn[constIndex];
#endif

        //staging for both parts

         idx[globalIndexCalc * DIM + dimIndex] = idxShared[threadLocalId]; //yupinov fix indexing
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
            dtheta_shared[thetaOffsetBase] = -data[0];

#pragma unroll
            for (int k = 1; k < order; k++)
            {
                dtheta_shared[thetaOffsetBase + k * thetaStride] = data[k - 1] - data[k];
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
                theta_shared[thetaOffsetBase + k * thetaStride] = data[k];
            }

            //yupinov store to global
            const int thetaGlobalOffsetBase = globalParticleIndexBase * DIM * order;
#pragma unroll
            for (int k = 0; k < order; k++)
            {
                const int thetaIndex = thetaOffsetBase + k * thetaStride;

                theta[thetaGlobalOffsetBase + thetaIndex] = theta_shared[thetaIndex];
                dtheta[thetaGlobalOffsetBase + thetaIndex] = dtheta_shared[thetaIndex];
            }
        }
    }
}


template <const int order, const int particlesPerBlock>
__global__ void pme_spread_kernel
(int start_ix, int start_iy, int start_iz,
  const int pny, const int pnz,
 const real * __restrict__ coefficientGlobal,
 real * __restrict__ grid, real * __restrict__ theta, const int * __restrict__ idx, //yupinov
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

    __shared__ int idxShared[thetaStride];
    __shared__ real coefficient[particlesPerBlock];

    __shared__ real theta_shared[thetaStride * order];
    //printf("%d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);

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
        idxShared[localIndexCalc * DIM + dimIndex] = idx[globalIndexCalc * DIM + dimIndex]; //yupinov check instructions

        const int thetaOffsetBase = localIndexCalc * DIM + dimIndex;
        const int thetaGlobalOffsetBase = globalParticleIndexBase * DIM * order;
#pragma unroll
        for (int k = 0; k < order; k++)
        {
            const int thetaIndex = thetaOffsetBase + k * thetaStride;
            theta_shared[thetaIndex] = theta[thetaGlobalOffsetBase + thetaIndex];
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
        const int ix = idxShared[localIndex * DIM + XX] - offx;
        const int iy = idxShared[localIndex * DIM + YY] - offy;
        const int iz = idxShared[localIndex * DIM + ZZ] - offz;

        const int thetaOffsetBase = localIndex * DIM;
        const real thz = theta_shared[thetaOffsetBase + ithz * thetaStride + ZZ];
        const real thy = theta_shared[thetaOffsetBase + ithy * thetaStride + YY];
        const real constVal = thz * thy * coefficient[localIndex];
        const int constOffset = (iy + ithy) * pnz + (iz + ithz);
        const real *thx = theta_shared + (thetaOffsetBase + XX);

#pragma unroll
        for (ithx = 0; (ithx < order); ithx++)
        {
            index_x = (ix + ithx) * pny * pnz;
            atomicAdd(grid + index_x + constOffset, thx[ithx * thetaStride] * constVal);
        }
    }
}

template <
    const int order
    >
__global__ void pme_wrap_kernel
    (const int nx, const int ny, const int nz,
     const int pny, const int pnz,
     real * __restrict__ grid
     )
{
    // const int overlap = order - 1;

    // WRAP
    int blockId = blockIdx.x
                 + blockIdx.y * gridDim.x
                 + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                  + (threadIdx.z * (blockDim.x * blockDim.y))
                  + (threadIdx.y * blockDim.x)
                  + threadIdx.x;

    //should use ldg.128

    if (threadId < OVERLAP_CELLS_COUNTS[OVERLAP_ZONES - 1])
    {   
        int zoneIndex = -1;
        do
        {
            zoneIndex++;
        }
        while (threadId >= OVERLAP_CELLS_COUNTS[zoneIndex]);
        const int2 zoneSizeYZ = OVERLAP_SIZES[zoneIndex];
        // this is the overlapped cells's index relative to the current zone
        const int cellIndex = (zoneIndex > 0) ? (threadId - OVERLAP_CELLS_COUNTS[zoneIndex - 1]) : threadId;

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

void spread_on_grid_lines_gpu(struct gmx_pme_t *pme, pme_atomcomm_t *atc,
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

    const int thread = 0;
    //yupinov
    // bCalcSplines is always true - untested, unfinished
    // bDoSplines is always false - untested
    // bSpread is always true - untested, unfinished
    // check bClearF as well

    cudaError_t stat;
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
    int n_blocked = n;//(n + warp_size - 1) / warp_size * warp_size;
    int ndatatot = pnx * pny * pnz;
    int size_grid = ndatatot * sizeof(real);

    int size_order = order * n * sizeof(real);
    int size_order_dim = size_order * DIM;
    real *theta_d = PMEFetchRealArray(PME_ID_THETA, thread, size_order_dim, ML_DEVICE);
    real *dtheta_d = PMEFetchRealArray(PME_ID_DTHETA, thread, size_order_dim, ML_DEVICE);

    // IDXPTR
    int idx_size = n * DIM * sizeof(int);
    int *idx_d = PMEFetchIntegerArray(PME_ID_IDXPTR, thread, idx_size, ML_DEVICE);

    real *fsh_d = NULL;
    int *nn_d = NULL;
    float3 *xptr_d = NULL;
    //float4 *xptr_d = NULL;

    const float3 nXYZ = {(real)nx, (real)ny, (real)nz};
    const int3 nnOffset = {0, 5 * nx, 5 * (nx + ny)};

    if (bCalcSplines)
    {
        const int fshSize = 5 * (nx + ny + nz) * sizeof(real);
        fsh_d = PMEFetchRealArray(PME_ID_FSH, thread, fshSize, ML_DEVICE);
        PMECopy(fsh_d                , pme->fshx, 5 * nx * sizeof(real), ML_DEVICE, s);
        PMECopy(fsh_d + 5 * nx       , pme->fshy, 5 * ny * sizeof(real), ML_DEVICE, s);
        PMECopy(fsh_d + 5 * (nx + ny), pme->fshz, 5 * nz * sizeof(real), ML_DEVICE, s);

        const int nnSize = 5 * (nx + ny + nz) * sizeof(int);
        nn_d = PMEFetchIntegerArray(PME_ID_NN, thread, nnSize, ML_DEVICE);
        PMECopy(nn_d                , pme->nnx, 5 * nx * sizeof(int), ML_DEVICE, s);
        PMECopy(nn_d + 5 * nx       , pme->nny, 5 * ny * sizeof(int), ML_DEVICE, s);
        PMECopy(nn_d + 5 * (nx + ny), pme->nnz, 5 * nz * sizeof(int), ML_DEVICE, s);

#if USE_TEXTURES
#if USE_TEXOBJ
        //if (use_texobj(dev_info))
        // commented texture object code - too lazy to check device info here for CC >= 3.0
        {
            cudaResourceDesc rd;
            cudaTextureDesc td;

            memset(&rd, 0, sizeof(rd));
            rd.resType                  = cudaResourceTypeLinear;
            rd.res.linear.devPtr        = fsh_d;
            rd.res.linear.desc.f        = cudaChannelFormatKindFloat;
            rd.res.linear.desc.x        = 32;
            rd.res.linear.sizeInBytes   = fshSize;
            memset(&td, 0, sizeof(td));
            td.readMode                 = cudaReadModeElementType;
            stat = cudaCreateTextureObject(&fshTexture, &rd, &td, NULL);
            CU_RET_ERR(stat, "cudaCreateTextureObject on fsh_d failed");


            memset(&rd, 0, sizeof(rd));
            rd.resType                  = cudaResourceTypeLinear;
            rd.res.linear.devPtr        = nn_d;
            rd.res.linear.desc.f        = cudaChannelFormatKindSigned;
            rd.res.linear.desc.x        = 32;
            rd.res.linear.sizeInBytes   = nnSize;
            memset(&td, 0, sizeof(td));
            td.readMode                 = cudaReadModeElementType;
            stat = cudaCreateTextureObject(&nnTexture, &rd, &td, NULL); //yupinov destroy, keep allocated
            CU_RET_ERR(stat, "cudaCreateTextureObject on nn_d failed");
        }
        else
#endif
        {
            cudaChannelFormatDesc cd_fsh = cudaCreateChannelDesc<float>();
            stat = cudaBindTexture(NULL, &fshTextureRef, fsh_d, &cd_fsh, fshSize);
            CU_RET_ERR(stat, "cudaBindTexture on fsh failed");

            cudaChannelFormatDesc cd_nn = cudaCreateChannelDesc<int>();
            stat = cudaBindTexture(NULL, &nnTextureRef,nn_d, &cd_nn, nnSize);
            CU_RET_ERR(stat, "cudaBindTexture on nn failed");

        }
#endif


        float3 *xptr_h = (float3 *)atc->x;
        xptr_d = (float3 *)PMEFetchRealArray(PME_ID_XPTR, thread, DIM * n_blocked * sizeof(real), ML_DEVICE);
        PMECopy(xptr_d, xptr_h, DIM * n_blocked * sizeof(real), ML_DEVICE, s);
        /*
        float4 *xptr_h = (float4 *)PMEFetchRealArray(PME_ID_XPTR, thread, 4 * n_blocked * sizeof(real), ML_HOST);
        memset(xptr_h, 0, 4 * n_blocked * sizeof(real));
        for (int i = 0; i < n; i++)
        {
           memcpy(xptr_h + i, atc->x + i, sizeof(rvec));
        }
        xptr_d = (float4 *)PMEFetchRealArray(PME_ID_XPTR, thread, 4 * n_blocked * sizeof(real), ML_DEVICE);
        PMECopy(xptr_d, xptr_h, 4 * n_blocked * sizeof(real), ML_DEVICE, s);
        */

        const float3 recipbox_h[3] =
        {
            {pme->recipbox[XX][XX], pme->recipbox[YY][XX], pme->recipbox[ZZ][XX]},
            {                  0.0, pme->recipbox[YY][YY], pme->recipbox[ZZ][YY]},
            {                  0.0,                   0.0, pme->recipbox[ZZ][ZZ]}
        };
        PMECopyConstant(RECIPBOX, recipbox_h, sizeof(recipbox_h), s);
    }


    //yupinov blocked approach everywhere or nowhere
    //filtering?

    real *coefficient_d = PMEFetchAndCopyRealArray(PME_ID_COEFFICIENT, thread, atc->coefficient, n * sizeof(real), ML_DEVICE, s); //yupinov compact here as weel?

    real *grid_d = NULL;
    if (bSpread)
    {
        grid_d = PMEFetchRealArray(PME_ID_REAL_GRID, thread, size_grid, ML_DEVICE);
        stat = cudaMemsetAsync(grid_d, 0, size_grid, s); //yupinov
        CU_RET_ERR(stat, "cudaMemsetAsync spread error");
    }
    /*
    const int N = 256;
    const int D = 2;
    int n_blocks = (n + N - 1) / N;
    dim3 dimGrid(n_blocks, 1, 1);
    dim3 dimBlock(order, order, D);
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
                    events_record_start(gpu_events_spline, s);

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
                                                                                                    nn_d, fsh_d,
#endif
                                                                                                    xptr_d,
                                                                                                    coefficient_d,
                                                                                                    theta_d, dtheta_d, idx_d,
                                                                                                    n);


                    }

                    CU_LAUNCH_ERR("pme_spline_kernel");

                    events_record_stop(gpu_events_spline, s, ewcsPME_SPLINE, 0);
                }
                if (bSpread)
                {
                    events_record_start(gpu_events_spread, s);

                    pme_spread_kernel<4, blockSize / 4 / 4> <<<nBlocks, dimBlock, 0, s>>>
                                                                            (pme->pmegrid_start_ix, pme->pmegrid_start_iy, pme->pmegrid_start_iz,
                                                                             pny, pnz,
                                                                             coefficient_d,
                                                                             grid_d, theta_d, idx_d,
                                                                             n);

                    CU_LAUNCH_ERR("pme_spread_kernel");

                    events_record_stop(gpu_events_spread, s, ewcsPME_SPREAD, 0);
                }
            }
            else // a single monster kernel here
            {
                events_record_start(gpu_events_splineandspread, s);

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
                                                                                                                               nn_d, fsh_d,
#endif
                                                                                                                               xptr_d,
                                                                                                                               coefficient_d,
                                                                                                                               grid_d, theta_d, dtheta_d, idx_d,
                                                                                                                               n);
                        }
                        else
                            gmx_fatal(FARGS, "the code for bSpread==false was not tested!");
                    }
                }
                else
                    gmx_fatal(FARGS, "the code for bCalcSplines==false was not tested!"); //yupinov
                CU_LAUNCH_ERR("pme_spline_and_spread_kernel");

                events_record_stop(gpu_events_splineandspread, s, ewcsPME_SPLINEANDSPREAD, 0);
            }
            if (bSpread && pme->bGPUSingle)
            {
                // wrap on GPU as a separate small kernel - we need a complete grid first!
                const int blockSize = 4 * warp_size; //yupinov this is everywhere! and arichitecture-specific

                // cell count in 7 parts of overlap
                const int3 zoneSizes_h[OVERLAP_ZONES] =
                {
                    {     nx,        ny,   overlap},
                    {     nx,   overlap,        nz},
                    {overlap,        ny,        nz},
                    {     nx,   overlap,   overlap},
                    {overlap,        ny,   overlap},
                    {overlap,   overlap,        nz},
                    {overlap,   overlap,   overlap}
                };

                const int2 zoneSizesYZ_h[OVERLAP_ZONES] =
                {
                    {     ny,   overlap},
                    {overlap,        nz},
                    {     ny,        nz},
                    {overlap,   overlap},
                    {     ny,   overlap},
                    {overlap,        nz},
                    {overlap,   overlap}
                };

                int cellsAccumCount_h[OVERLAP_ZONES];
                for (int i = 0; i < OVERLAP_ZONES; i++)
                    cellsAccumCount_h[i] = zoneSizes_h[i].x * zoneSizes_h[i].y * zoneSizes_h[i].z;
                // accumulate
                for (int i = 1; i < OVERLAP_ZONES; i++)
                {
                    cellsAccumCount_h[i] = cellsAccumCount_h[i] + cellsAccumCount_h[i - 1];
                }

                const int overlappedCells = (nx + overlap) * (ny + overlap) * (nz + overlap) - nx * ny * nz;
                const int nBlocks = (overlappedCells + blockSize - 1) / blockSize;

                PMECopyConstant(OVERLAP_SIZES, zoneSizesYZ_h, sizeof(zoneSizesYZ_h), s);
                PMECopyConstant(OVERLAP_CELLS_COUNTS, cellsAccumCount_h, sizeof(cellsAccumCount_h), s);
                //other constants

                events_record_start(gpu_events_wrap, s);

                pme_wrap_kernel<4> <<<nBlocks, blockSize, 0, s>>>(nx, ny, nz, pny, pnz, grid_d);

                CU_LAUNCH_ERR("pme_wrap_kernel");

                events_record_stop(gpu_events_wrap, s, ewcsPME_WRAP, 0);
            }
            break;

        default:
            gmx_fatal(FARGS, "the code for pme_order != 4 was not tested!"); //yupinov
    }

    if (!pme->gpu->keepGPUDataBetweenSpreadAndR2C)
    {
        if (bSpread)
            PMECopy(pmegrid->grid, grid_d, size_grid, ML_HOST, s);
        for (int j = 0; j < DIM; ++j) //also breaking compacting in gather
        //and why not just check bGPUSingle here?
        {
            PMECopy(atc->spline[thread].dtheta[j], dtheta_d + j * n * order, size_order, ML_HOST, s);
            PMECopy(atc->spline[thread].theta[j], theta_d + j * n * order, size_order, ML_HOST, s);
        }
        //yupinov what about theta/dtheta/idx use in pme_realloc_atomcomm_things?
        PMECopy(atc->idx, idx_d, idx_size, ML_HOST, s);
    }
    //yupinov check flags like bSpread etc. before copying...

    //yupinov free, keep allocated
    /*
    cudaFree(theta_d);
    cudaFree(dtheta_d);
    cudaFree(fractx_d);
    cudaFree(coefficient_d);
    free(fractx_h);
    free(coefficient_h);
    */
}

