#include "pme.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include <cuda.h>

#include "th-a.cuh"
#include "check.h"

#include "pme-internal.h"
#include "pme-cuda.h"

typedef real *splinevec[DIM];
#ifdef DEBUG_PME_GPU
extern gpu_flags gather_gpu_flags;
#endif
#ifdef DEBUG_PME_TIMINGS_GPU
extern gpu_events gpu_events_gather;
#endif

#define DO_FSPLINE(order)                      \
    for (int ithx = 0; (ithx < order); ithx++)              \
    {                                              \
        int index_x = (i0[i]+ithx)*pny*pnz;               \
        real tx      = thx[iorder+ithx];                       \
        real dx      = dthx[iorder+ithx];                      \
                                               \
        for (int ithy = 0; (ithy < order); ithy++)          \
        {                                          \
            int index_xy = index_x+(j0[i]+ithy)*pnz;      \
            real ty       = thy[iorder+ithy];                  \
            real dy       = dthy[iorder+ithy];                 \
            real fxy1     = 0, fz1 = 0;		   \
                                               \
            for (int ithz = 0; (ithz < order); ithz++)      \
            {                                      \
                /*printf(" INDEX %d %d %d\n", (i0[i] + ithx), (j0[i]+ithy), (k0[i]+ithz));*/\
                real gval  = grid[index_xy+(k0[i]+ithz)];  \
                fxy1 += thz[iorder+ithz]*gval;            \
                fz1  += dthz[iorder+ithz]*gval;           \
            }                                      \
            fx += dx*ty*fxy1;                      \
            fy += tx*dy*fxy1;                      \
            fz += tx*ty*fz1;                       \
        }                                          \
    }


static __global__ void gather_f_bsplines_kernel
(real *grid, int order, int n,
 int nx, int ny, int nz, int pnx, int pny, int pnz,
 real rxx, real ryx, real ryy, real rzx, real rzy, real rzz,
 real *thx, real *thy, real *thz, real *dthx, real *dthy, real *dthz,
 real *atc_f, real *coefficient_v, int *i0, int *j0, int *k0)
{
  /* sum forces for local particles */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    real coefficient = coefficient_v[i];
    real fx     = 0;
    real fy     = 0;
    real fz     = 0;
    int iorder = i*order;
    int idim = i * DIM;

    switch (order)
    {
    case 4:
      DO_FSPLINE(4);
      break;
    case 5:
      DO_FSPLINE(5);
      break;
    default:
      DO_FSPLINE(order);
      break;
    }


    atc_f[idim + XX] += -coefficient*( fx*nx*rxx );
    atc_f[idim + YY] += -coefficient*( fx*nx*ryx + fy*ny*ryy );
    atc_f[idim + ZZ] += -coefficient*( fx*nx*rzx + fy*ny*rzy + fz*nz*rzz );

    /*printf("kernel coeff=%f f=%f,%f,%f\n",
	   (double) coefficient,
	   (double) fx, (double) fy, (double) fz);*/

    /* Since the energy and not forces are interpolated
     * the net force might not be exactly zero.
     * This can be solved by also interpolating F, but
     * that comes at a cost.
     * A better hack is to remove the net force every
     * step, but that must be done at a higher level
     * since this routine doesn't see all atoms if running
     * in parallel. Don't know how important it is?  EL 990726
     */
  }
}

void gather_f_bsplines_gpu_2_pre
(gmx_bool bClearF,
 int *spline_ind, int spline_n,
 real *atc_coefficient, rvec *atc_f,
 real scale, int thread
 )
{
    // compact atc_f before cpu calcucation

    int size_forces = DIM * spline_n * sizeof(real);
    real *atc_f_compacted = th_a(TH_ID_F, thread, size_forces, TH_LOC_HOST); //yupinov fixed allocation size - not actually compacted, same for i_compacted
    int size_indices = spline_n * sizeof(int);
    int *atc_i_compacted = th_i(TH_ID_I, thread, size_indices, TH_LOC_HOST);

    int oo = 0;
    for (int ii = 0; ii < spline_n; ii++)
    {
        int i           = spline_ind[ii];
        real coefficient_i = scale*atc_coefficient[i];
        if (bClearF)
        {
            atc_f[i][XX] = 0;
            atc_f[i][YY] = 0;
            atc_f[i][ZZ] = 0;
        }

        if (coefficient_i != 0.0)
        {
            atc_f_compacted[oo * DIM + XX] = atc_f[i][XX];
            atc_f_compacted[oo * DIM + YY] = atc_f[i][YY];
            atc_f_compacted[oo * DIM + ZZ] = atc_f[i][ZZ];
            atc_i_compacted[oo] = i;  // indices of uncompacted particles stored in a compacted array
            oo++;
        }
    }
    //oo is a real size of compacted stuff now
}

void gather_f_bsplines_gpu_2
(real *grid, gmx_bool bClearF,
 int order,
 int nx, int ny, int nz, int pnx, int pny, int pnz,
 real rxx, real ryx, real ryy, real rzx, real rzy, real rzz,
 int *spline_ind, int spline_n,
 real *atc_coefficient, rvec *atc_f, ivec *atc_idx,
 splinevec *spline_theta, splinevec *spline_dtheta,
 real scale,
 gmx_pme_t *pme,
 int thread
 )
{
    cudaStream_t s = pme->gpu->pmeStream;
    int ndatatot = pnx*pny*pnz;

    if (!spline_n)
        return;

    int size_grid = ndatatot * sizeof(real);
    real *grid_d = th_a_cpy(TH_ID_GRID, thread, grid, size_grid, TH_LOC_CUDA, s);

    //copy order?
    //compacting, and size....
    int n = spline_n;
    int size_indices = n * sizeof(int);
    int size_coefficients = n * sizeof(real);
    int size_forces = DIM * n * sizeof(real);
    int size_splines = order * n * sizeof(int);


    real *atc_f_compacted = th_a(TH_ID_F, thread, -1, TH_LOC_HOST); //but that's wrong! realloc

    int *atc_i_compacted = th_i(TH_ID_I, thread, -1, TH_LOC_HOST);  //way to get sizes from th-a?
    real *coefficients_compacted = th_a(TH_ID_COEFFICIENT, thread, size_coefficients, TH_LOC_HOST);
    //yupinov reuse H_ID_COEFFICIENT and other stuff from before solve?

    int *i0_compacted = th_i(TH_ID_I0, thread, size_indices, TH_LOC_HOST); //yupinov these are IDXPTR, actually. maybe split it?
    int *j0_compacted = th_i(TH_ID_J0, thread, size_indices, TH_LOC_HOST);
    int *k0_compacted = th_i(TH_ID_K0, thread, size_indices, TH_LOC_HOST);

    real *theta_x_compacted = th_a(TH_ID_THX, thread, size_splines, TH_LOC_HOST);
    real *theta_y_compacted = th_a(TH_ID_THY, thread, size_splines, TH_LOC_HOST);
    real *theta_z_compacted = th_a(TH_ID_THZ, thread, size_splines, TH_LOC_HOST);
    real *dtheta_x_compacted = th_a(TH_ID_DTHX, thread, size_splines, TH_LOC_HOST);
    real *dtheta_y_compacted = th_a(TH_ID_DTHY, thread, size_splines, TH_LOC_HOST);
    real *dtheta_z_compacted = th_a(TH_ID_DTHZ, thread, size_splines, TH_LOC_HOST);

    int oo = 0;
    for (int ii = 0; ii < spline_n; ii++)
    {
        int i           = spline_ind[ii];
        real coefficient_i = scale*atc_coefficient[i];
        if (bClearF)
        {
            atc_f[i][XX] = 0; //yupinov memeset?
            atc_f[i][YY] = 0;
            atc_f[i][ZZ] = 0;
        }

        if (coefficient_i != 0)
        {
            coefficients_compacted[oo] = coefficient_i;
            int *idxptr = atc_idx[i];
            //Mattias: atc_f_h force-copying is in gather_f_bsplines_gpu_2_pre()
            //yupinov: the fuck is it doing there?
            atc_i_compacted[oo] = i;
            i0_compacted[oo] = idxptr[XX];
            j0_compacted[oo] = idxptr[YY];
            k0_compacted[oo] = idxptr[ZZ];
            int iiorder = ii*order;
            int ooorder = oo*order;
            for (int o = 0; o < order; ++o)
            {
                theta_x_compacted[ooorder + o] = (*spline_theta)[XX][iiorder + o];
                theta_y_compacted[ooorder + o] = (*spline_theta)[YY][iiorder + o];
                theta_z_compacted[ooorder + o] = (*spline_theta)[ZZ][iiorder + o];
                dtheta_x_compacted[ooorder + o] = (*spline_dtheta)[XX][iiorder + o];
                dtheta_y_compacted[ooorder + o] = (*spline_dtheta)[YY][iiorder + o];
                dtheta_z_compacted[ooorder + o] = (*spline_dtheta)[ZZ][iiorder + o];
            }
            ++oo;
        }
    }

    n = oo;
    if (!n)
        return;

    //copypasted
    size_indices = n * sizeof(int);
    size_coefficients = n * sizeof(real);
    size_forces = DIM * n * sizeof(real);
    size_splines = order * n * sizeof(int);

    real *atc_f_d = th_a_cpy(TH_ID_F, thread, atc_f_compacted, size_forces, TH_LOC_CUDA, s);
    real *coefficients_d = th_a_cpy(TH_ID_COEFFICIENT, thread, coefficients_compacted, size_coefficients, TH_LOC_CUDA, s);

    int *i0_d = th_i_cpy(TH_ID_I0, thread, i0_compacted, size_indices, TH_LOC_CUDA, s);
    int *j0_d = th_i_cpy(TH_ID_J0, thread, j0_compacted, size_indices, TH_LOC_CUDA, s);
    int *k0_d = th_i_cpy(TH_ID_K0, thread, k0_compacted, size_indices, TH_LOC_CUDA, s);

    real *theta_x_d = th_a_cpy(TH_ID_THX, thread, theta_x_compacted, size_splines, TH_LOC_CUDA, s);
    real *theta_y_d = th_a_cpy(TH_ID_THY, thread, theta_y_compacted, size_splines, TH_LOC_CUDA, s);
    real *theta_z_d = th_a_cpy(TH_ID_THZ, thread, theta_z_compacted, size_splines, TH_LOC_CUDA, s);
    real *dtheta_x_d = th_a_cpy(TH_ID_DTHX, thread, dtheta_x_compacted, size_splines, TH_LOC_CUDA, s);
    real *dtheta_y_d = th_a_cpy(TH_ID_DTHY, thread, dtheta_y_compacted, size_splines, TH_LOC_CUDA, s);
    real *dtheta_z_d = th_a_cpy(TH_ID_DTHZ, thread, dtheta_z_compacted, size_splines, TH_LOC_CUDA, s);

    int block_size = 2 * warp_size;
    int n_blocks = (n + block_size - 1) / block_size;
#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_start(gpu_events_gather, s);
#endif
    gather_f_bsplines_kernel<<<n_blocks, block_size, 0, s>>>
      (grid_d,
       order, n,
       nx, ny, nz, pnx, pny, pnz,
       rxx, ryx, ryy, rzx, rzy, rzz,
       theta_x_d, theta_y_d, theta_z_d,
       dtheta_x_d, dtheta_y_d, dtheta_z_d,
       atc_f_d, coefficients_d,
       i0_d, j0_d, k0_d);
    CU_LAUNCH_ERR("gather_f_bsplines_kernel");
#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_stop(gpu_events_gather, s, ewcsPME_GATHER, 0);
#endif

    th_cpy(atc_f_compacted, atc_f_d, size_forces, TH_LOC_HOST, s);

    for (int ii = 0; ii < n; ii++)  // iterating over compacted particles
    {
        int i = atc_i_compacted[ii]; //index of uncompacted particle
        atc_f[i][XX] = atc_f_compacted[ii * DIM + XX];
        atc_f[i][YY] = atc_f_compacted[ii * DIM + YY];
        atc_f[i][ZZ] = atc_f_compacted[ii * DIM + ZZ];
    }
}
