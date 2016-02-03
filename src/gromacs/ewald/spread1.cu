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

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/math/vectypes.h"
#include "check.h"

#include <cuda_runtime.h>

#define PME_ORDER_MAX 12
typedef real *splinevec[DIM];
#ifdef DEBUG_PME_GPU
extern gpu_flags spread_gpu_flags;
extern gpu_flags spread_bunching_gpu_flags;
extern gpu_events gpu_events_spread;
#endif

#include "thread_mpi/mutex.h"

/* This has to be a macro to enable full compiler optimization with xlC (and probably others too) */
#define DO_BSPLINE(order)                            \


#define SPREAD_COEFFICIENTS_KERNEL(order) \

template <int order>
__global__ void spread1_coefficients_kernel_O(int n,
					     real *grid,
					     int *i0, int *j0, int *k0,
					     int pny, int pnz,
					     real *coefficient,
					     real *thx, real *thy, real *thz)
{
  int ithz = threadIdx.x;
  int ithy = threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;
  if (i < n) {
    if (coefficient[i]) {
      _Pragma("unroll")
      for (int ithx = 0; ithx < order; ithx++)
      {
	int index_x = (i0[i]+ithx)*pny*pnz;
	real valx    = coefficient[i]*thx[i*order+ithx];

	real valxy    = valx*thy[i*order+ithy];
	int index_xy = index_x+(j0[i]+ithy)*pnz;

	int index_xyz        = index_xy+(k0[i]+ithz);
	/*grid[index_xyz] += valxy*thz[i*order+ithz];*/
	atomicAdd(&grid[index_xyz], valxy*thz[i*order+ithz]);
      }
    }
  }
}

#include "th-a.cuh"

static tMPI::mutex print_mutex;

void spread1_coefficients_bsplines_thread_gpu_2
(int pnx, int pny, int pnz, int offx, int offy, int offz,
 real *grid, int order, ivec *atc_idx, int *spline_ind, int spline_n,
 real *atc_coefficient, splinevec *spline_theta, int atc_n_foo,
 int thread)
{
  //fprintf(stderr, "Hello spread! %d %d\n", thread, spline_n);

    int ndatatot = pnx*pny*pnz;
    int size_grid = ndatatot * sizeof(real);
#ifdef DEBUG_PME_GPU
    real *grid_check;
    if (check_vs_cpu_j(spread_gpu_flags, 1)) {
      grid_check = th_a(TH_ID_GRID, thread, size_grid, TH_LOC_HOST);
      memcpy(grid_check, grid, ndatatot * sizeof(real));
    }
#endif
    for (int i = 0; i < ndatatot; i++)
    {
      // FIX clear grid on device instead
        grid[i] = 0;
    }

    real *grid_d = th_a(TH_ID_GRID, thread, size_grid, TH_LOC_CUDA);
    cudaMemcpy(grid_d, grid, size_grid, cudaMemcpyHostToDevice);

    int size_real = spline_n * sizeof(real);
    int size_int = spline_n * sizeof(int);
    int *i0 = th_i(TH_ID_I0, thread, size_int, TH_LOC_HOST);
    int *j0 = th_i(TH_ID_J0, thread, size_int, TH_LOC_HOST);
    int *k0 = th_i(TH_ID_K0, thread, size_int, TH_LOC_HOST);
    real *coefficient = th_a(TH_ID_COEFFICIENT, thread, size_real, TH_LOC_HOST);
    real *thx = th_a(TH_ID_THX, thread, size_real * order, TH_LOC_HOST);
    real *thy = th_a(TH_ID_THY, thread, size_real * order, TH_LOC_HOST);
    real *thz = th_a(TH_ID_THZ, thread, size_real * order, TH_LOC_HOST);

    int *i0_d = th_i(TH_ID_I0, thread, size_int, TH_LOC_CUDA);
    int *j0_d = th_i(TH_ID_J0, thread, size_int, TH_LOC_CUDA);
    int *k0_d = th_i(TH_ID_K0, thread, size_int, TH_LOC_CUDA);
    real *coefficient_d = th_a(TH_ID_COEFFICIENT, thread, size_real, TH_LOC_CUDA);
    real *thx_d = th_a(TH_ID_THX, thread, size_real * order, TH_LOC_CUDA);
    real *thy_d = th_a(TH_ID_THY, thread, size_real * order, TH_LOC_CUDA);
    real *thz_d = th_a(TH_ID_THZ, thread, size_real * order, TH_LOC_CUDA);

    int oo = 0;

    for (int ii = 0; ii < spline_n; ii++)
    {
        int i           = spline_ind[ii];
        real coefficient_i = atc_coefficient[i];
	//if (coefficient_i == 0) {
	//   continue;
	//}

	coefficient[oo] = coefficient_i;

	int *idxptr = atc_idx[i];
	int iiorder = ii*order;
	int ooorder = oo*order;

	i0[oo]   = idxptr[XX] - offx;
	j0[oo]   = idxptr[YY] - offy;
	k0[oo]   = idxptr[ZZ] - offz;

	for (int o = 0; o < order; ++o) {
	  thx[ooorder + o] = (*spline_theta)[XX][iiorder + o];
	  thy[ooorder + o] = (*spline_theta)[YY][iiorder + o];
	  thz[ooorder + o] = (*spline_theta)[ZZ][iiorder + o];
	}
	++oo;
    }

    int n = oo;

    //fprintf(stderr, "World! %d %d/%d\n", thread, n, spline_n);

    cudaMemcpy(i0_d, i0, size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(j0_d, j0, size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(k0_d, k0, size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(coefficient_d, coefficient, size_real, cudaMemcpyHostToDevice);
    cudaMemcpy(thx_d, thx, size_real * order, cudaMemcpyHostToDevice);
    cudaMemcpy(thy_d, thy, size_real * order, cudaMemcpyHostToDevice);
    cudaMemcpy(thz_d, thz, size_real * order, cudaMemcpyHostToDevice);

  int block_size = 32;
  int n_blocks = (n + block_size - 1) / block_size;
  dim3 dimGrid(1, 1, n_blocks);
  dim3 dimBlockOrder(order, order, block_size);
  dim3 dimBlockOne(1, 1, block_size);
  #ifdef DEBUG_PME_GPU
  events_record_start(gpu_events_spread);
#endif
    switch (order)
    {
    case 4: spread1_coefficients_kernel_O<4><<<dimGrid, dimBlockOrder>>>
	(n, grid_d, i0_d, j0_d, k0_d, pny, pnz,
	 coefficient_d, thx_d, thy_d, thz_d); break;
    case 5: spread1_coefficients_kernel_O<5><<<dimGrid, dimBlockOrder>>>
	(n, grid_d, i0_d, j0_d, k0_d, pny, pnz,
	 coefficient_d, thx_d, thy_d, thz_d); break;
    default: /* FIXME */ break;
    }
    #ifdef DEBUG_PME_GPU
  events_record_stop(gpu_events_spread, ewcsPME_SPREAD, 1);

    if (check_vs_cpu_j(spread_gpu_flags, 1))
    {
        print_mutex.lock(); //yupinov mutex - multilevel?
        fprintf(stderr, "Check %d  (%d x %d x %d)\n", thread, pnx, pny, pnz);
        print_mutex.unlock();
        for (int i = 0; i < ndatatot; i+=pnz)
            check_real(NULL, &grid_d[i], &grid_check[i], pnz, true, true);
    }
#endif
    cudaMemcpy(grid, grid_d, size_grid, cudaMemcpyDeviceToHost);
}
