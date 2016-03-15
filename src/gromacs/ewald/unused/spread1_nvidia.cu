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
#include "gromacs/gpu_utils/cudautils.cuh"

#include <cuda_runtime.h>

#define PME_ORDER_MAX 12
typedef real *splinevec[DIM];
#ifdef DEBUG_PME_TIMINGS_GPU
extern gpu_events gpu_events_spread;
#endif
#include "thread_mpi/mutex.h"

/* This has to be a macro to enable full compiler optimization with xlC (and probably others too) */
#define DO_BSPLINE(order)                            \


#define SPREAD_COEFFICIENTS_KERNEL(order) \

template <int order,int particles_per_block>
__global__ void spread1_coefficients_kernel_O(
                         const int n,
                         real * __restrict__ const grid,
                         const int * __restrict__ const i0,
                         const int * __restrict__ const j0,
                         const int * __restrict__ const k0,
                         const int pny, const int pnz,
                         const real * __restrict__ const coefficient,
                         const real * __restrict__ const thx,
                         const real * __restrict__ const thy,
                         const real * __restrict__ const thz)
{
  __shared__ real thx_shared[order*particles_per_block];
  __shared__ real thy_shared[order*particles_per_block];
  __shared__ real thz_shared[order*particles_per_block];
  __shared__ real coefficient_shared[particles_per_block];
  __shared__ int i0_shared[particles_per_block];
  __shared__ int j0_shared[particles_per_block];
  __shared__ int k0_shared[particles_per_block];
  
  for( int i_base = blockIdx.x*particles_per_block; i_base < n; i_base+=gridDim.x*particles_per_block )
  {
      const int local_idx = threadIdx.z *(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
      if ( local_idx < particles_per_block && (i_base + local_idx) < n )
      {
        coefficient_shared[local_idx] = coefficient[i_base + local_idx];
        i0_shared[local_idx] = i0[i_base + local_idx];
        j0_shared[local_idx] = j0[i_base + local_idx];
        k0_shared[local_idx] = k0[i_base + local_idx];
      }
      const int th_idx = i_base*order + local_idx;
      if ( th_idx - 0*particles_per_block*order < n*order &&
           local_idx >= 0*particles_per_block*order &&
           local_idx <  1*particles_per_block*order )
      {
        thx_shared[local_idx-0*particles_per_block*order]
            = thx[th_idx-0*particles_per_block*order];
      }
      if ( th_idx - 1*particles_per_block*order < n*order &&
           local_idx >= 1*particles_per_block*order &&
           local_idx <  2*particles_per_block*order )
      {
        thy_shared[local_idx-1*particles_per_block*order]
            = thy[th_idx-1*particles_per_block*order];
      }
      if ( th_idx - 2*particles_per_block*order < n*order &&
           local_idx >= 2*particles_per_block*order &&
           local_idx <  3*particles_per_block*order )
      {
        thz_shared[local_idx-2*particles_per_block*order]
            = thz[th_idx-2*particles_per_block*order];
      }
      __syncthreads();
      
      const int ithz = threadIdx.x;
      const int ithy = threadIdx.y;
      
      //assert( ithx < order && ithy < order && ithz < order );
      {
          const int i = i_base + threadIdx.z;
          if (i < n)
          {
            const real coeff = coefficient_shared[threadIdx.z];
            if (coeff)
            {
                const int th_offset = threadIdx.z*order;
                
                const real thy_val = thy_shared[th_offset+ithy];
                const real thz_val = thz_shared[th_offset+ithz];
                
                const int i0i = i0_shared[threadIdx.z];
                const int j0i_ithy = j0_shared[threadIdx.z] + ithy;
                const int k0i_ithz = k0_shared[threadIdx.z] + ithz;

                #pragma unroll
                for (int ithx = 0; ithx < order; ++ithx)
                {
                    const int index_x = (i0i+ithx)*(pny*pnz);
                    const real valx    = coeff*thx_shared[th_offset+ithx];

                    const real valxy    = valx*thy_val;
                    const int index_xy = index_x+(j0i_ithy)*pnz;
                    
                    const int index_xyz        = index_xy+k0i_ithz;
                    atomicAdd(&grid[index_xyz], valxy*thz_val);
                }
            }
          }
      }
  }
}

#include "th-a.cuh"

static tMPI::mutex print_mutex;

void spread1_nvidia_coefficients_bsplines_thread_gpu_2
(int pnx, int pny, int pnz, int offx, int offy, int offz,
 real *grid, int order, ivec *atc_idx, int *spline_ind, int spline_n,
 real *atc_coefficient, splinevec *spline_theta, int atc_n_foo,
 int thread)
{
  //fprintf(stderr, "Hello spread! %d %d\n", thread, spline_n);

    int ndatatot = pnx*pny*pnz;
    int size_grid = ndatatot * sizeof(real);
    for (int i = 0; i < ndatatot; i++)
    {
      // FIX clear grid on device instead
        grid[i] = 0;
    }

    real *grid_d = th_a(PME_ID_GRID, thread, size_grid, TH_LOC_CUDA);
    cudaMemcpy(grid_d, grid, size_grid, cudaMemcpyHostToDevice);

    int size_real = spline_n * sizeof(real);
    int size_int = spline_n * sizeof(int);
    int *i0 = th_i(PME_ID_I0, thread, size_int, TH_LOC_HOST);
    int *j0 = th_i(PME_ID_J0, thread, size_int, TH_LOC_HOST);
    int *k0 = th_i(PME_ID_K0, thread, size_int, TH_LOC_HOST);
    real *coefficient = th_a(PME_ID_COEFFICIENT, thread, size_real, TH_LOC_HOST);
    real *thx = th_a(PME_ID_THX, thread, size_real * order, TH_LOC_HOST);
    real *thy = th_a(PME_ID_THY, thread, size_real * order, TH_LOC_HOST);
    real *thz = th_a(PME_ID_THZ, thread, size_real * order, TH_LOC_HOST);

    int *i0_d = th_i(PME_ID_I0, thread, size_int, TH_LOC_CUDA);
    int *j0_d = th_i(PME_ID_J0, thread, size_int, TH_LOC_CUDA);
    int *k0_d = th_i(PME_ID_K0, thread, size_int, TH_LOC_CUDA);
    real *coefficient_d = th_a(PME_ID_COEFFICIENT, thread, size_real, TH_LOC_CUDA);
    real *thx_d = th_a(PME_ID_THX, thread, size_real * order, TH_LOC_CUDA);
    real *thy_d = th_a(PME_ID_THY, thread, size_real * order, TH_LOC_CUDA);
    real *thz_d = th_a(PME_ID_THZ, thread, size_real * order, TH_LOC_CUDA);

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

  const int particles_per_block = 32;
  int n_blocks = (n + particles_per_block - 1) / particles_per_block;
  dim3 dimGrid(n_blocks, 1, 1);
  dim3 dimBlockOrder(order, order, particles_per_block);
#ifdef DEBUG_PME_TIMINGS_GPU
  events_record_start(gpu_events_spread);
#endif
    switch (order)
    {
    case 4: spread1_coefficients_kernel_O<4,particles_per_block><<<dimGrid, dimBlockOrder>>>
	(n, grid_d, i0_d, j0_d, k0_d, pny, pnz,
	 coefficient_d, thx_d, thy_d, thz_d); break;
    case 5: spread1_coefficients_kernel_O<5,particles_per_block><<<dimGrid, dimBlockOrder>>>
	(n, grid_d, i0_d, j0_d, k0_d, pny, pnz,
	 coefficient_d, thx_d, thy_d, thz_d); break;
    default: /* FIXME */ break;
    }
    CU_LAUNCH_ERR("spread1_nvidia_coefficients_kernel");

#ifdef DEBUG_PME_TIMINGS_GPU
  events_record_stop(gpu_events_spread, ewcsPME_SPREAD, 1);
#endif
  cudaMemcpy(grid, grid_d, size_grid, cudaMemcpyDeviceToHost);
}
