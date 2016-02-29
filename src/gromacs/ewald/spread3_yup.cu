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
//yupinov unused file!
#include "pme.h"
#include "pme-internal.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/math/vectypes.h"
#include "check.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include <cuda_runtime.h>

typedef real *splinevec[DIM];
#ifdef DEBUG_PME_GPU
extern gpu_flags spread_gpu_flags;
extern gpu_flags spread_bunching_gpu_flags;
#endif
#ifdef DEBUG_PME_TIMINGS_GPU
extern gpu_events gpu_events_spread;
#endif
#include "thread_mpi/mutex.h"

#include "th-a.cuh"

static tMPI::mutex print_mutex;



/* This has to be a macro to enable full compiler optimization with xlC (and probably others too) */

#define DO_BSPLINE(order)                                         \
    _Pragma("unroll")                                                    \
    if (globalParticleIndex == 0) \
        printf("%g coefficient\n", coefficient[globalParticleIndex]); \
    for (ithx = 0; (ithx < order); ithx++)                    \
    {                                                             \
        index_x = (i0 + ithx) * pny * pnz;                    \
        valx = coefficient[globalParticleIndex] * thx[ithx];                      \
        if (globalParticleIndex == 0) \
        {    printf("%d index_x\n", index_x); \
         printf("%d %d %d %d\n", i0, ithx, pny, pnz);} \
        _Pragma("unroll")                                                         \
        for (ithy = 0; (ithy < order); ithy++)                \
        {                                                         \
            valxy    = valx*thy[ithy];                       \
            index_xy = index_x+(j0+ithy)*pnz;                 \
             _Pragma("unroll")                                                     \
            for (ithz = 0; (ithz < order); ithz++)            \
            {                                                     \
                index_xyz        = index_xy+(k0+ithz);        \
                /*grid[index_xyz] += valxy*thz[ithz];*/               \
                atomicAdd(grid + index_xyz, valxy*thz[ithz]);      \
                if (globalParticleIndex == 0) \
                    printf("contributing %g to %i\n", valxy*thz[ithz], index_xyz); \
            }                                                     \
        }                                                         \
    }




//template <int order, int N, int K, int D>
// K is particles per block?
template <int order, int particlesPerBlock>
__global__ void spread3_kernel
(int nx, int ny, int nz,
 int start_ix, int start_iy, int start_iz,
 real rxx, real ryx, real ryy, real rzx, real rzy, real rzz,
 //int *g2tx, int *g2ty, int *g2tz,
 real *fshx, real *fshy,
 int *nnx, int *nny, int *nnz,
 real *xptr, real *yptr, real *zptr,
 real *coefficient,
 real *grid,
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

    const int offx = 0, offy = 0, offz = 0;
    const int pnx = nx + order - 1, pny = ny + order - 1, pnz = nz + order - 1; //yupinov fix me!

    //const int B = K / D / order / order;

    __shared__ int idxxptr[particlesPerBlock];
    __shared__ int idxyptr[particlesPerBlock];
    __shared__ int idxzptr[particlesPerBlock];
    __shared__ real fxptr[particlesPerBlock];
    __shared__ real fyptr[particlesPerBlock];
    __shared__ real fzptr[particlesPerBlock];

    __shared__ real theta[3 * order * particlesPerBlock];
    __shared__ real dtheta[3 * order * particlesPerBlock];
    //printf("%d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);

    // so I have particlesPerBlock to process with warp_size threads?

    int ithx, index_x, ithy, index_xy, ithz, index_xyz;
    real valx, valxy;


    int localParticleIndex = threadIdx.x;  //yupinov
    int globalParticleIndex = blockIdx.x * particlesPerBlock + localParticleIndex;
    if (globalParticleIndex < n)
    //yupinov - this is a single particle work!
    {
        // INTERPOL_IDX

        /* Fractional coordinates along box vectors, add 2.0 to make 100% sure we are positive for triclinic boxes */
        real tx, ty, tz;
        tx = nx * ( xptr[globalParticleIndex] * rxx + yptr[globalParticleIndex] * ryx + zptr[globalParticleIndex] * rzx + 2.0 );
        ty = ny * (                                   yptr[globalParticleIndex] * ryy + zptr[globalParticleIndex] * rzy + 2.0 );
        tz = nz * (                                                                     zptr[globalParticleIndex] * rzz + 2.0 );

        int tix, tiy, tiz;
        tix = (int)(tx);
        tiy = (int)(ty);
        tiz = (int)(tz);
        if (globalParticleIndex == 0)
        for (int rr = 0; rr < 5 * nx; rr++)
            printf("%d test %d\n", rr, nnx[rr]);
        if (globalParticleIndex == 0)
            printf("%d %d tix\n", tix, nnx[tix]);
        /* Because decomposition only occurs in x and y,
        * we never have a fraction correction in z.
        */

        fxptr[localParticleIndex] = tx - tix + fshx[tix];
        fyptr[localParticleIndex] = ty - tiy + fshy[tiy];
        fzptr[localParticleIndex] = tz - tiz;

        idxxptr[localParticleIndex] = nnx[tix];
        idxyptr[localParticleIndex] = nny[tiy];
        idxzptr[localParticleIndex] = nnz[tiz];

        // CALCSPLINE

        if (coefficient[globalParticleIndex] != 0.0) //yupinov weak
        {
            real dr, div;
            real data[order];

            _Pragma("unroll")
            for (int j = 0; j < DIM; j++)
            {
                //dr  = fractx[i*DIM + j];
                dr = j == 0 ? fxptr[localParticleIndex] : (j == 1 ? fyptr[localParticleIndex] : fzptr[localParticleIndex]);

                /* dr is relative offset from lower cell limit */
                data[order - 1] = 0;
                data[1]         = dr;
                data[0]         = 1 - dr;

                _Pragma("unroll")
                for (int k = 3; k < order; k++)
                {
                    div         = 1.0 / (k - 1.0);
                    data[k - 1] = div * dr * data[k - 2];
                    _Pragma("unroll")
                    for (int l = 1; l < (k - 1); l++)
                    {
                        data[k - l - 1] = div * ((dr + l) * data[k - l - 2] + (k - l - dr) * data[k - l - 1]);
                    }
                    data[0] = div * (1 - dr) * data[0];
                }
                /* differentiate */
                int thetaOffset = (j * particlesPerBlock + localParticleIndex) * order;
                dtheta[thetaOffset] = -data[0];

                _Pragma("unroll")
                for (int k = 1; k < order; k++)
                {
                    dtheta[thetaOffset + k] = data[k - 1] - data[k];
                }

                div             = 1.0 / (order - 1);
                data[order - 1] = div * dr * data[order - 2];
                _Pragma("unroll")
                for (int l = 1; l < (order - 1); l++)
                {
                    data[order - l - 1] = div * ((dr + l) * data[order - l - 2] + (order - l - dr) * data[order - l - 1]);
                }
                data[0] = div * (1 - dr) * data[0];

                _Pragma("unroll")
                for (int k = 0; k < order; k++)
                {
                    theta[thetaOffset + k] = data[k];
                }
            }
        }


        if (coefficient[globalParticleIndex] != 0.0) //yupinov weak
        {
            int i0   = idxxptr[localParticleIndex] - offx; //?
            int j0   = idxyptr[localParticleIndex] - offy;
            int k0   = idxzptr[localParticleIndex] - offz;

            real *thx = theta + (0 * particlesPerBlock + localParticleIndex) * order;
            real *thy = theta + (1 * particlesPerBlock + localParticleIndex) * order;
            real *thz = theta + (2 * particlesPerBlock + localParticleIndex) * order;

           // switch (order)
            DO_BSPLINE(order);
        }
    }
    //yupinov start spread copypaste



      //yupinov end spread copypaste





    /*
#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_stop(gpu_events_calcspline, ewcsPME_CALCSPLINE, 0);
#endif
    */


    //yupinov
    /*
    cudaFree(theta_d);
    cudaFree(dtheta_d);
    cudaFree(fractx_d);
    cudaFree(coefficient_d);
    free(fractx_h);
    free(coefficient_h);
    */




    //int block_i0 = blockIdx.x * N;
    //int idxspline_i = (threadIdx.z * order + threadIdx.y) * order + threadIdx.x; // 0 -> 31 -> index of first or second particle contrib....
/*
    for (int block_i = 0; block_i < N; block_i += K) //256, += 32 => each iter is a warp
    {
        //int i = block_i0 + block_i + idxspline_i; //block_i0 + block_i is a particle index
        //if ((blockIdx.x == 1) && (idxspline_i == 1))
        //    printf("%d %d %d %d %d\n", blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z, i);
        //so why why would you add thread id?
       // if (i < N)  //it is always less than N, isn't it?
        {
            //printf("%d %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, i);
            ;

        }*/
    /*
        if (particlesPerBlock > 32) //???
        {
            __syncthreads();
        }
        */
        //printf("hello2 %d\n", idxspline_i);
        // SPREAD
        /*
        for (int spread_i0 = 0; spread_i0 < K; spread_i0 += B) // B is 1 now = stride?
        {
            int i = block_i0 + block_i + spread_i0 + threadIdx.z / D;
            if (i < n)
            {
                int *i0 = idxxptr;
                int *j0 = idxyptr;
                int *k0 = idxzptr;
                real *thx = theta;
                real *thy = thx + order*N;
                real *thz = thy + order*N;

                int ithz = threadIdx.x;
                int ithy = threadIdx.y;
                int i = blockIdx.z * blockDim.z + threadIdx.z;
                if (i < n)
                {
                    if (coefficient[i])
                    {
                        _Pragma("unroll")
                        for (int ithx0 = 0; ithx0 < order; ithx0 += D)
                        {
                            int ithx = ithx0 + threadIdx.z % D;
                            int index_x = (i0[i]+ithx)*ny*nz;
                            real valx    = coefficient[i]*thx[i*order+ithx];

                            real valxy    = valx*thy[i*order+ithy];
                            int index_xy = index_x+(j0[i]+ithy)*nz;

                            int index_xyz        = index_xy+(k0[i]+ithz);
                            //grid[index_xyz] += valxy*thz[i*order+ithz];
                            atomicAdd(&grid[index_xyz], valxy*thz[i*order+ithz]);
                        }
                    }
                }
            }
        }
        */
    //}
}


void spread3_yup_gpu(struct gmx_pme_t *pme, pme_atomcomm_t *atc,
         int grid_index,
         pmegrid_t *pmegrid)
{
  int nx = pme->nkx, ny = pme->nky, nz = pme->nkz;
  //int nx = pmegrid->s[XX], ny = pmegrid->s[YY], nz = pmegrid->s[ZZ];
  real *grid = pmegrid->grid;
  const int order = pmegrid->order;
  int thread = 0;


    printf("%d second\n", pme->nnx[201]);

  int n = atc->n;
  int n_blocked = (n + warp_size - 1) / warp_size * warp_size;
  int ndatatot = nx*ny*nz;
  int size_grid = ndatatot * sizeof(real);

#ifdef DEBUG_PME_GPU
  // GRID CHECK


  real *grid_check;
  if (check_vs_cpu_j(spread_gpu_flags, 3))
  {
    grid_check = th_a(TH_ID_GRID, thread, size_grid, TH_LOC_HOST);
    memcpy(grid_check, pmegrid, ndatatot * sizeof(real));
  }
#endif
  // G2T
  /*
  int *g2tx_h = pme->pmegrid[grid_index].g2t[XX];
  int *g2ty_h = pme->pmegrid[grid_index].g2t[YY];
  int *g2tz_h = pme->pmegrid[grid_index].g2t[ZZ];
  int *g2tx_d = th_i(TH_ID_G2T, thread, 3 * n32 * sizeof(int), TH_LOC_CUDA);
  int *g2ty_d = g2tx_d + n32;
  int *g2tz_d = g2ty_d + n32;
  cudaMemcpy(g2tx_d, g2tx_h, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g2ty_d, g2ty_h, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g2tz_d, g2tz_h, n * sizeof(int), cudaMemcpyHostToDevice);
  */

  // FSH
  real *fshx_d = th_a(TH_ID_FSH, thread, 5 * (nx + ny) * sizeof(real), TH_LOC_CUDA);
  real *fshy_d = fshx_d + 5 * nx;
  cudaMemcpy(fshx_d, pme->fshx, 5 * nx * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(fshy_d, pme->fshy, 5 * ny * sizeof(real), cudaMemcpyHostToDevice);

  // NN
  int *nnx_d = th_i(TH_ID_NN, thread, 5 * (nx + ny + nz) * sizeof(int), TH_LOC_CUDA);
  int *nny_d = nnx_d + 5 * nx;
  int *nnz_d = nny_d + 5 * ny;
  cudaMemcpy(nnx_d, pme->nnx, 5 * nx * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nny_d, pme->nny, 5 * ny * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nnz_d, pme->nnz, 5 * nz * sizeof(int), cudaMemcpyHostToDevice);

  // XPTR
  real *xptr_h = th_a(TH_ID_XPTR, thread, 3 * n_blocked * sizeof(real), TH_LOC_HOST);
  real *xptr_d = th_a(TH_ID_XPTR, thread, 3 * n_blocked * sizeof(real), TH_LOC_CUDA);
  real *yptr_d = xptr_d + n_blocked;
  real *zptr_d = yptr_d + n_blocked;
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
  cudaMemcpy(xptr_d, xptr_h, 3 * n_blocked * sizeof(real), cudaMemcpyHostToDevice);

  // COEFFICIENT
  real *coefficient_d = th_a(TH_ID_COEFFICIENT, thread, n * sizeof(real), TH_LOC_CUDA);
  cudaMemcpy(coefficient_d, atc->coefficient, n * sizeof(real), cudaMemcpyHostToDevice);

  // GRID

  for (int i = 0; i < ndatatot; i++)
    {
      // FIX clear grid on device instead
      grid[i] = 0;
    }

  cudaError_t stat;
  real *grid_d = th_a(TH_ID_GRID, thread, size_grid, TH_LOC_CUDA);
  //stat = cudaMemcpy(grid_d, grid, size_grid, cudaMemcpyHostToDevice);
  //CU_RET_ERR(stat, "cudaMemcpy spread error");
  stat = cudaMemset(grid_d, 0, size_grid);
  CU_RET_ERR(stat, "cudaMemset spread error");
      printf("%d third\n", pme->nnx[201]);
#ifdef DEBUG_PME_TIMINGS_GPU
  events_record_start(gpu_events_spread);
#endif
  /*
  const int N = 256;
  const int D = 2;
  int n_blocks = (n + N - 1) / N;
  dim3 dimGrid(n_blocks, 1, 1);
  dim3 dimBlock(order, order, D);
  */
  const int particlesPerBlock = warp_size;
  //const int D = 2;
  dim3 nBlocks((n + particlesPerBlock - 1) / particlesPerBlock, 1, 1);
  //dim3 dimBlock(order, order, D); //each block has 32 threads now to hand 32 particlesPerBlock
  dim3 dimBlock(particlesPerBlock, 1, 1); //yupinov heavy
  switch (order)
  {
  case 4:
      /*
    const int O = 4;
    const int B = 1;
    const int K = B * D * O * O;
    */
    //spread3_kernel<4, N, K, D><<<dimGrid, dimBlock>>>
      spread3_kernel<4, particlesPerBlock><<<nBlocks, dimBlock>>>
      (nx, ny, nz,
       pme->pmegrid_start_ix, pme->pmegrid_start_iy, pme->pmegrid_start_iz,
       pme->recipbox[XX][XX],
       pme->recipbox[YY][XX],
       pme->recipbox[YY][YY],
       pme->recipbox[ZZ][XX],
       pme->recipbox[ZZ][YY],
       pme->recipbox[ZZ][ZZ],
       //g2tx_d, g2ty_d, g2tz_d,
       fshx_d, fshy_d,
       nnx_d, nny_d, nnz_d,
       xptr_d, yptr_d, zptr_d,
       coefficient_d,
       grid_d,
       n);
    //yupinov orders
  }
  CU_LAUNCH_ERR("spread3_kernel");

#ifdef DEBUG_PME_TIMINGS_GPU
  events_record_stop(gpu_events_spread, ewcsPME_SPREAD, 3);
#endif
#ifdef DEBUG_PME_GPU
  if (check_vs_cpu_j(spread_gpu_flags, 3)) {
    print_mutex.lock();
    fprintf(stderr, "Check %d  (%d x %d x %d)\n",
        thread, nx, ny, nz);
    for (int i = 0; i < ndatatot; ++i) {
      real diff = grid_check[i];
      real cpu_v = grid_check[i];
      cudaMemcpy(&grid_check[i], &grid_d[i], sizeof(real), cudaMemcpyDeviceToHost);
      diff -= grid_check[i];
      real gpu_v = grid_check[i];
      if (diff != 0) {
    real absdiff = fabs(diff) / fabs(cpu_v);
    if (absdiff > .000001) {
      fprintf(stderr, "%dppm", (int) (absdiff * 1e6));
      if (absdiff > .0001) {
        fprintf(stderr, " value %f ", cpu_v);
      }
    } else {
      fprintf(stderr, "~");
    }
    //fprintf(stderr, "(%f - %f)", cpu_v, gpu_v);
      } else {
    if (gpu_v == 0) {
      fprintf(stderr, "0");
    } else {
      fprintf(stderr, "=");
    }
      }
      if ((i + 1) % nz == 0) {
    fprintf(stderr, "\n");
      }
    }
    print_mutex.unlock();
  }
  #endif
  stat = cudaMemcpy(grid, grid_d, size_grid, cudaMemcpyDeviceToHost); //yupinov part of check?
  CU_RET_ERR(stat, "cudaMemcpy spread3 error");
}

