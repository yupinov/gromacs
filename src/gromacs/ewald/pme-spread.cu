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

// wrap kernel thingies - should be kept in pme-cuda.h as common?
static const int OVERLAP_ZONES = 7;
__constant__ __device__ int OVERLAP_CELLS_COUNTS[OVERLAP_ZONES];
__constant__ __device__ int2 OVERLAP_SIZES[OVERLAP_ZONES];

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
(int nx, int ny, int nz,
 int start_ix, int start_iy, int start_iz,
 const int pny, const int pnz,
 real rxx, real ryx, real ryy, real rzx, real rzy, real rzz,
 const real * __restrict__ fshx, const real * __restrict__ fshy,
 const int * __restrict__ nnx, const int * __restrict__ nny, const int * __restrict__ nnz,
 const real * __restrict__ xptr, const real * __restrict__ yptr, const real * __restrict__ zptr,
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

    __shared__ int idxxptr[particlesPerBlock];
    __shared__ int idxyptr[particlesPerBlock];
    __shared__ int idxzptr[particlesPerBlock];
    __shared__ real fxptr[particlesPerBlock];
    __shared__ real fyptr[particlesPerBlock];
    __shared__ real fzptr[particlesPerBlock];
    __shared__ real coefficient[particlesPerBlock];

    __shared__ real theta_shared[3 * order * particlesPerBlock];
    __shared__ real dtheta_shared[3 * order * particlesPerBlock];

    int ithx, index_x, ithy, index_xy, ithz, index_xyz;
    real valx, valxy;

    int localIndex = threadIdx.x;

    int globalIndex = blockIdx.x * particlesPerBlock + localIndex;
    if (bCalcSplines)
    {
        if ((globalIndex < n) && (threadIdx.y == 0) && (threadIdx.z == 0)) //yupinov - the stupid way of just multiplying the thread number by order^2 => particlesPerBlock==2 threads do this in every block
        //yupinov - this is a single particle work!
        {
            // INTERPOLATION INDICES

            /* Fractional coordinates along box vectors, add 2.0 to make 100% sure we are positive for triclinic boxes */
            real tx, ty, tz;
            tx = nx * ( xptr[globalIndex] * rxx + yptr[globalIndex] * ryx + zptr[globalIndex] * rzx + 2.0f );
            ty = ny * (                           yptr[globalIndex] * ryy + zptr[globalIndex] * rzy + 2.0f );
            tz = nz * (                                                     zptr[globalIndex] * rzz + 2.0f );

            int tix, tiy, tiz;
            tix = (int)(tx);
            tiy = (int)(ty);
            tiz = (int)(tz);
            /* Because decomposition only occurs in x and y,
            * we never have a fraction correction in z.
            */

            fxptr[localIndex] = tx - tix + fshx[tix];
            fyptr[localIndex] = ty - tiy + fshy[tiy];
            fzptr[localIndex] = tz - tiz;

            idxxptr[localIndex] = nnx[tix];
            idxyptr[localIndex] = nny[tiy];
            idxzptr[localIndex] = nnz[tiz];

            coefficient[localIndex] = coefficientGlobal[globalIndex]; //staging for both parts

            idx[globalIndex * DIM + 0] = idxxptr[localIndex];
            idx[globalIndex * DIM + 1] = idxyptr[localIndex];
            idx[globalIndex * DIM + 2] = idxzptr[localIndex];

            // MAKE BSPLINES

            if (bDoSplines || (coefficient[localIndex] != 0.0f)) //yupinov how bad is this conditional?
            {
                real dr, div;
                real data[order];

    #pragma unroll
                for (int j = 0; j < DIM; j++)
                {
                    //dr  = fractx[i*DIM + j];

                    dr = (j == 0) ? fxptr[localIndex] : ((j == 1) ? fyptr[localIndex] : fzptr[localIndex]);

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
                    const int thetaOffset = (j * particlesPerBlock + localIndex) * order;
                    dtheta_shared[thetaOffset] = -data[0];

                    #pragma unroll
                    for (int k = 1; k < order; k++)
                    {
                        dtheta_shared[thetaOffset + k] = data[k - 1] - data[k];
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
                        theta_shared[thetaOffset + k] = data[k];
                    }
                }
                //yupinov store to global
    #pragma unroll
                for (int j = 0; j < DIM; j++)
                {
                    const int thetaOffset = (j * particlesPerBlock + localIndex) * order;
                    const int thetaGlobalOffset = (j * n + globalIndex) * order;
    #pragma unroll
                    for (int z = 0; z < order; z++)
                    {
                        theta[thetaGlobalOffset + z] = theta_shared[thetaOffset + z];
                        dtheta[thetaGlobalOffset + z] = dtheta_shared[thetaOffset + z];
                    }
                }
            }
        }
        __syncthreads(); //yupinov do we need it?
    }
    else if (bSpread) //yupinov copied from spread_kernel - staging for a second part
    {
        if ((globalIndex < n) && (threadIdx.y == 0) && (threadIdx.z == 0)) //yupinov - the stupid way of just multiplying the thread number by order^2 => particlesPerBlock==2 threads do this in every block
        //yupinov - this is a single particle work!
        {
            idxxptr[localIndex] = idx[globalIndex * DIM + 0];
            idxyptr[localIndex] = idx[globalIndex * DIM + 1];
            idxzptr[localIndex] = idx[globalIndex * DIM + 2];
    #pragma unroll
            for (int j = 0; j < DIM; j++)
            {
                const int thetaOffset = (j * particlesPerBlock + localIndex) * order;
                const int thetaGlobalOffset = (j * n + globalIndex) * order;
    #pragma unroll
                for (int z = 0; z < order; z++)
                {
                    theta_shared[thetaOffset + z] = theta[thetaGlobalOffset + z];
                    dtheta_shared[thetaOffset + z] = dtheta[thetaGlobalOffset + z];
                }
            }

            coefficient[localIndex] = coefficientGlobal[globalIndex];
        }
        __syncthreads(); //yupinov do we need it?
    }


    // SPREAD

    // spline Y/Z coordinates
    ithy = threadIdx.y;
    ithz = threadIdx.z;

    if (bSpread)
    {
        if ((globalIndex < n) && (coefficient[localIndex] != 0.0f)) //yupinov store checks
        {
            const int i0  = idxxptr[localIndex] - offx; //?
            const int j0  = idxyptr[localIndex] - offy;
            const int k0  = idxzptr[localIndex] - offz;

            //printf ("%d %d %d %d %d %d %d\n", blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z, i0, j0, k0);
            const real *thx = theta_shared + (0 * particlesPerBlock + localIndex) * order;
            const real *thy = theta_shared + (1 * particlesPerBlock + localIndex) * order;
            const real *thz = theta_shared + (2 * particlesPerBlock + localIndex) * order;

            // switch (order) ?

            #pragma unroll
            for (ithx = 0; (ithx < order); ithx++)
            {
                index_x = (i0 + ithx) * pny * pnz;
                valx = coefficient[localIndex] * thx[ithx];
                /*
                #pragma unroll
                for (ithy = 0; (ithy < order); ithy++)
                */
                {
                    valxy    = valx*thy[ithy];
                    index_xy = index_x+(j0+ithy)*pnz;
                    /*
                    #pragma unroll
                    for (ithz = 0; (ithz < order); ithz++)
                    */
                    {
                        index_xyz        = index_xy + (k0 + ithz);
                        atomicAdd(grid + index_xyz, valxy * thz[ithz]);
                    }
                }
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
(const int nx, const int ny, const int nz,
 const int start_ix, const int start_iy, const int start_iz,
 const real rxx, const real ryx, const real ryy, const real rzx, const real rzy, const real rzz,
 const real * __restrict__ fshx, const real * __restrict__ fshy,
 const int * __restrict__ nnx, const int * __restrict__ nny, const int * __restrict__ nnz,
 const real * __restrict__ xptr, const real * __restrict__ yptr, const real * __restrict__ zptr,
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

    const int offx = 0, offy = 0, offz = 0;//yupinov fix me!

    __shared__ int idxxptr[particlesPerBlock];
    __shared__ int idxyptr[particlesPerBlock];
    __shared__ int idxzptr[particlesPerBlock];
    __shared__ real fxptr[particlesPerBlock];
    __shared__ real fyptr[particlesPerBlock];
    __shared__ real fzptr[particlesPerBlock];
    __shared__ real coefficient[particlesPerBlock];

    __shared__ real theta_shared[3 * order * particlesPerBlock];
    __shared__ real dtheta_shared[3 * order * particlesPerBlock];

    int localIndex = threadIdx.x;

    int globalIndex = blockIdx.x * particlesPerBlock + localIndex;
    if ((globalIndex < n) && (threadIdx.y == 0) && (threadIdx.z == 0)) //yupinov - the stupid way of just multiplying the thread number by order^2 => particlesPerBlock==2 threads do this in every block
    //yupinov - this is a single particle work!
    {
        // INTERPOL_IDX

        /* Fractional coordinates along box vectors, add 2.0 Fto make 100% sure we are positive for triclinic boxes */
        real tx, ty, tz;
        tx = nx * ( xptr[globalIndex] * rxx + yptr[globalIndex] * ryx + zptr[globalIndex] * rzx + 2.0f );
        ty = ny * (                           yptr[globalIndex] * ryy + zptr[globalIndex] * rzy + 2.0f );
        tz = nz * (                                                     zptr[globalIndex] * rzz + 2.0f );

        int tix, tiy, tiz;
        tix = (int)(tx);
        tiy = (int)(ty);
        tiz = (int)(tz);
        /* Because decomposition only occurs in x and y,
        * we never have a fraction correction in z.
        */

        fxptr[localIndex] = tx - tix + fshx[tix];
        fyptr[localIndex] = ty - tiy + fshy[tiy];
        fzptr[localIndex] = tz - tiz;

        idxxptr[localIndex] = nnx[tix];
        idxyptr[localIndex] = nny[tiy];
        idxzptr[localIndex] = nnz[tiz];

        coefficient[localIndex] = coefficientGlobal[globalIndex]; //staging for both parts


        idx[globalIndex * DIM + 0] = idxxptr[localIndex];
        idx[globalIndex * DIM + 1] = idxyptr[localIndex];
        idx[globalIndex * DIM + 2] = idxzptr[localIndex];

        // CALCSPLINE

        if (bDoSplines || (coefficient[localIndex] != 0.0f)) //yupinov how bad is this conditional?
        {
            real dr, div;
            real data[order];

#pragma unroll
            for (int j = 0; j < DIM; j++)
            {
                //dr  = fractx[i*DIM + j];

                dr = (j == 0) ? fxptr[localIndex] : ((j == 1) ? fyptr[localIndex] : fzptr[localIndex]);

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
                const int thetaOffset = (j * particlesPerBlock + localIndex) * order;
                dtheta_shared[thetaOffset] = -data[0];

                #pragma unroll
                for (int k = 1; k < order; k++)
                {
                    dtheta_shared[thetaOffset + k] = data[k - 1] - data[k];
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
                    theta_shared[thetaOffset + k] = data[k];
                }
            }
            //yupinov store to global
#pragma unroll
            for (int j = 0; j < DIM; j++)
            {
                const int thetaOffset = (j * particlesPerBlock + localIndex) * order;
                const int thetaGlobalOffset = (j * n + globalIndex) * order;
#pragma unroll
                for (int z = 0; z < order; z++)
                {
                    theta[thetaGlobalOffset + z] = theta_shared[thetaOffset + z];
                    dtheta[thetaGlobalOffset + z] = dtheta_shared[thetaOffset + z];
                }
            }
        }
    }
}


template <const int order, const int particlesPerBlock>
__global__ void pme_spread_kernel
(int nx, int ny, int nz,
 int start_ix, int start_iy, int start_iz,
  const int pny, const int pnz,
 const real * __restrict__ coefficientGlobal,
 real * __restrict__ grid, real * __restrict__ theta, real * __restrict__ dtheta, const int * __restrict__ idx, //yupinov
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

    const int offx = 0, offy = 0, offz = 0;//yupinov fix me!

    __shared__ int idxxptr[particlesPerBlock];
    __shared__ int idxyptr[particlesPerBlock];
    __shared__ int idxzptr[particlesPerBlock];
    __shared__ real coefficient[particlesPerBlock];

    __shared__ real theta_shared[3 * order * particlesPerBlock];
    __shared__ real dtheta_shared[3 * order * particlesPerBlock];
    //printf("%d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);

    int ithx, index_x, ithy, index_xy, ithz, index_xyz;
    real valx, valxy;

    const int localIndex = threadIdx.x;
    const int globalIndex = blockIdx.x * particlesPerBlock + localIndex;

    if ((globalIndex < n) && (threadIdx.y == 0) && (threadIdx.z == 0)) //yupinov - the stupid way of just multiplying the thread number by order^2 => particlesPerBlock==2 threads do this in every block
    //yupinov - this is a single particle work!
    {

        idxxptr[localIndex] = idx[globalIndex * DIM + 0];
        idxyptr[localIndex] = idx[globalIndex * DIM + 1];
        idxzptr[localIndex] = idx[globalIndex * DIM + 2];
#pragma unroll
        for (int j = 0; j < DIM; j++)
        {
            const int thetaOffset = (j * particlesPerBlock + localIndex) * order;
            const int thetaGlobalOffset = (j * n + globalIndex) * order;
#pragma unroll
            for (int z = 0; z < order; z++)
            {
                theta_shared[thetaOffset + z] = theta[thetaGlobalOffset + z];
                dtheta_shared[thetaOffset + z] = dtheta[thetaGlobalOffset + z];
            }
        }

        coefficient[localIndex] = coefficientGlobal[globalIndex]; //staging for both parts
    }
    __syncthreads(); //yupinov do we need it?

    // SPREAD

    // spline Y/Z coordinates
    ithy = threadIdx.y;
    ithz = threadIdx.z;

    if ((globalIndex < n) && (coefficient[localIndex] != 0.0f)) //yupinov store checks
    {
        const int i0  = idxxptr[localIndex] - offx; //?
        const int j0  = idxyptr[localIndex] - offy;
        const int k0  = idxzptr[localIndex] - offz;

        //printf ("%d %d %d %d %d %d %d\n", blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z, i0, j0, k0);
        const real *thx = theta_shared + (0 * particlesPerBlock + localIndex) * order;
        const real *thy = theta_shared + (1 * particlesPerBlock + localIndex) * order;
        const real *thz = theta_shared + (2 * particlesPerBlock + localIndex) * order;

        // switch (order) ?

        #pragma unroll
        for (ithx = 0; (ithx < order); ithx++)
        {
            index_x = (i0 + ithx) * pny * pnz;
            valx = coefficient[localIndex] * thx[ithx];
            /*
            #pragma unroll
            for (ithy = 0; (ithy < order); ithy++)
            */
            {
                valxy    = valx*thy[ithy];
                index_xy = index_x+(j0+ithy)*pnz;
                /*
                #pragma unroll
                for (ithz = 0; (ithz < order); ithz++)
                */
                {
                    index_xyz        = index_xy + (k0 + ithz);
                    atomicAdd(grid + index_xyz, valxy * thz[ithz]);
                }
            }
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
    const gmx_bool separateKernels = false;
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

    ivec local_ndata, local_size, local_offset;
    /*
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
    int n_blocked = (n + warp_size - 1) / warp_size * warp_size;
    int ndatatot = pnx * pny * pnz;
    int size_grid = ndatatot * sizeof(real);

    int size_order = order * n * sizeof(real);
    int size_order_dim = size_order * DIM;
    real *theta_d = PMEFetchRealArray(PME_ID_THETA, thread, size_order_dim, ML_DEVICE);
    real *dtheta_d = PMEFetchRealArray(PME_ID_DTHETA, thread, size_order_dim, ML_DEVICE);

    // IDXPTR
    int idx_size = n * DIM * sizeof(int);
    int *idx_d = PMEFetchIntegerArray(PME_ID_IDXPTR, thread, idx_size, ML_DEVICE);

    real *fshx_d, *fshy_d = NULL;
    int *nnx_d = NULL, *nny_d = NULL, *nnz_d = NULL;
    real *xptr_d = NULL, *yptr_d = NULL, *zptr_d = NULL;

    if (bCalcSplines)
    {
        fshx_d = PMEFetchRealArray(PME_ID_FSH, thread, 5 * (nx + ny) * sizeof(real), ML_DEVICE);
        fshy_d = fshx_d + 5 * nx;
        PMECopy(fshx_d, pme->fshx, 5 * nx * sizeof(real), ML_DEVICE, s);
        PMECopy(fshy_d, pme->fshy, 5 * ny * sizeof(real), ML_DEVICE, s);

        nnx_d = PMEFetchIntegerArray(PME_ID_NN, thread, 5 * (nx + ny + nz) * sizeof(int), ML_DEVICE);
        nny_d = nnx_d + 5 * nx;
        nnz_d = nny_d + 5 * ny;
        PMECopy(nnx_d, pme->nnx, 5 * nx * sizeof(int), ML_DEVICE, s);
        PMECopy(nny_d, pme->nny, 5 * ny * sizeof(int), ML_DEVICE, s);
        PMECopy(nnz_d, pme->nnz, 5 * nz * sizeof(int), ML_DEVICE, s);

        real *xptr_h = PMEFetchRealArray(PME_ID_XPTR, thread, 3 * n_blocked * sizeof(real), ML_HOST);
        xptr_d = PMEFetchRealArray(PME_ID_XPTR, thread, 3 * n_blocked * sizeof(real), ML_DEVICE);
        yptr_d = xptr_d + n_blocked;
        zptr_d = yptr_d + n_blocked;
        {
            int ix = 0, iy = n_blocked, iz = 2 * n_blocked;
            for (int i = 0; i < n; i++)
            {
              real *xptr = atc->x[i];
              xptr_h[ix++] = xptr[XX];
              xptr_h[iy++] = xptr[YY];
              xptr_h[iz++] = xptr[ZZ];
            }
        }
        PMECopy(xptr_d, xptr_h, 3 * n_blocked * sizeof(real), ML_DEVICE, s);
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
                                                                            (nx, ny, nz,
                                                                             pme->pmegrid_start_ix, pme->pmegrid_start_iy, pme->pmegrid_start_iz,
                                                                             pme->recipbox[XX][XX],
                                                                             pme->recipbox[YY][XX],
                                                                             pme->recipbox[YY][YY],
                                                                             pme->recipbox[ZZ][XX],
                                                                             pme->recipbox[ZZ][YY],
                                                                             pme->recipbox[ZZ][ZZ],
                                                                             fshx_d, fshy_d,
                                                                             nnx_d, nny_d, nnz_d,
                                                                             xptr_d, yptr_d, zptr_d,
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
                                                                            (nx, ny, nz,
                                                                             pme->pmegrid_start_ix, pme->pmegrid_start_iy, pme->pmegrid_start_iz,
                                                                             pny, pnz,
                                                                             coefficient_d,
                                                                             grid_d, theta_d, dtheta_d, idx_d,
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
                                                                            (nx, ny, nz,
                                                                             pme->pmegrid_start_ix, pme->pmegrid_start_iy, pme->pmegrid_start_iz,
                                                                             pny, pnz,
                                                                             pme->recipbox[XX][XX],
                                                                             pme->recipbox[YY][XX],
                                                                             pme->recipbox[YY][YY],
                                                                             pme->recipbox[ZZ][XX],
                                                                             pme->recipbox[ZZ][YY],
                                                                             pme->recipbox[ZZ][ZZ],
                                                                             fshx_d, fshy_d,
                                                                             nnx_d, nny_d, nnz_d,
                                                                             xptr_d, yptr_d, zptr_d,
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

                stat = cudaMemcpyToSymbolAsync(OVERLAP_SIZES, zoneSizesYZ_h, sizeof(zoneSizesYZ_h), 0, cudaMemcpyHostToDevice, s);
                CU_RET_ERR(stat, "PME spread cudaMemcpyToSymbol");
                stat = cudaMemcpyToSymbolAsync(OVERLAP_CELLS_COUNTS, cellsAccumCount_h, sizeof(cellsAccumCount_h), 0, cudaMemcpyHostToDevice, s);
                CU_RET_ERR(stat, "PME spread cudaMemcpyToSymbol");
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
        } //yupinov what about theta/dtheta/idx use in pme_realloc_atomcomm_things?
    }
    PMECopy(atc->idx, idx_d, idx_size, ML_HOST, s);
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

