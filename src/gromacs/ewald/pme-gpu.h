#ifndef PMEGPU_H
#define PMEGPU_H

#include "pme-internal.h"
//#include "gromacs/fft/parallel_3dfft.h"
#include "gromacs/gpu_utils/gpu_macros.h"

//yupinov also lots of unused parameters warnings!

/*
#if GMX_GPU == GMX_GPU_OPENCL
struct gmx_nbnxn_ocl_t;
typedef struct gmx_nbnxn_ocl_t gmx_nbnxn_gpu_t;
#endif

#if GMX_GPU == GMX_GPU_CUDA
struct gmx_nbnxn_cuda_t;
typedef struct gmx_nbnxn_cuda_t gmx_nbnxn_gpu_t;
#endif

#if GMX_GPU == GMX_GPU_NONE
typedef int gmx_nbnxn_gpu_t;
#endif
*/

//yupinov add pmegpu detection warning!
//yupinov add author info everywhere
//yupinov CUDA_FUNC_QUALIFIER everywhere? as weel as parameters
// gmx_unused




CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_init_gpu(gmx_parallel_3dfft_gpu_t gmx_unused *pfft_setup,
                               ivec        gmx_unused              ndata,
                               real gmx_unused **real_data,
                               t_complex gmx_unused **complex_data,
                               MPI_Comm  gmx_unused                comm[2],
                               gmx_bool    gmx_unused              bReproducible,
                               int         gmx_unused              nthreads,
                                gmx_pme_t *pme)  CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_real_limits_gpu(gmx_parallel_3dfft_gpu_t gmx_unused pfft_setup,
                               ivec          gmx_unused            local_ndata,
                               ivec         gmx_unused             local_offset,
                               ivec         gmx_unused             local_size) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_complex_limits_gpu(gmx_parallel_3dfft_gpu_t gmx_unused pfft_setup,
                                  ivec     gmx_unused                 complex_order,
                                  ivec       gmx_unused               local_ndata,
                                  ivec       gmx_unused               local_offset,
                                  ivec       gmx_unused               local_size) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_execute_gpu(gmx_parallel_3dfft_gpu_t gmx_unused pfft_setup,
                           enum gmx_fft_direction gmx_unused dir,
                           int             gmx_unused        thread,
                           gmx_wallcycle_t   gmx_unused      wcycle,
                            gmx_pme_t *pme) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void calc_interpolation_idx_gpu_core
(int gmx_unused nx, int gmx_unused ny, int gmx_unused nz,
 real gmx_unused rxx, real gmx_unused ryx, real gmx_unused ryy, real gmx_unused rzx, real gmx_unused rzy, real gmx_unused rzz,
 real gmx_unused *fshx, real gmx_unused *fshy,
 int gmx_unused *nnx, int gmx_unused *nny, int gmx_unused *nnz,
 rvec gmx_unused *xptr_v, ivec gmx_unused *idxptr_v, rvec gmx_unused *fptr_v,
 int gmx_unused start, int gmx_unused end, int gmx_unused thread) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void make_bsplines_gpu(splinevec gmx_unused theta_v, splinevec gmx_unused dtheta_v, int gmx_unused order,
               rvec gmx_unused fractx_v[], int gmx_unused nr, int gmx_unused ind[], real gmx_unused coefficient[],
               gmx_bool gmx_unused bDoSplines, int gmx_unused thread) CUDA_FUNC_TERM //yupinov why thread (here, and in other _gpu functions?!

CUDA_FUNC_QUALIFIER void spread_coefficients_bsplines_thread_gpu_2
(int gmx_unused pnx, int gmx_unused pny, int gmx_unused pnz, int gmx_unused offx, int gmx_unused offy, int gmx_unused offz,
 real gmx_unused *grid, int gmx_unused order, ivec gmx_unused *atc_idx, int gmx_unused *spline_ind, int gmx_unused spline_n,
 real gmx_unused *atc_coefficient, splinevec gmx_unused *spline_theta, int gmx_unused atc_n_foo,
 int gmx_unused thread) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void spread1_coefficients_bsplines_thread_gpu_2
(int gmx_unused pnx, int gmx_unused pny, int gmx_unused pnz, int gmx_unused offx, int gmx_unused offy, int gmx_unused offz,
 real gmx_unused *grid, int gmx_unused order, ivec gmx_unused *atc_idx, int gmx_unused *spline_ind, int gmx_unused spline_n,
 real gmx_unused *atc_coefficient, splinevec gmx_unused *spline_theta, int gmx_unused atc_n_foo,
 int gmx_unused thread) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void spread1_nvidia_coefficients_bsplines_thread_gpu_2
(int gmx_unused pnx, int gmx_unused pny, int gmx_unused pnz, int gmx_unused offx, int gmx_unused offy, int gmx_unused offz,
 real gmx_unused *grid, int gmx_unused order, ivec gmx_unused *atc_idx, int gmx_unused *spline_ind, int gmx_unused spline_n,
 real gmx_unused *atc_coefficient, splinevec gmx_unused *spline_theta, int gmx_unused atc_n_foo,
 int gmx_unused thread) CUDA_FUNC_TERM

//yupinov inline?

CUDA_FUNC_QUALIFIER void spread2_coefficients_bsplines_thread_gpu_2
(int gmx_unused pnx, int gmx_unused pny, int gmx_unused pnz, int gmx_unused offx, int gmx_unused offy, int gmx_unused offz,
 real gmx_unused *grid, int gmx_unused order, ivec gmx_unused *atc_idx, int gmx_unused *spline_ind, int gmx_unused spline_n,
 real gmx_unused *atc_coefficie, splinevec gmx_unused *spline_theta, int gmx_unused atc_n_foo,
 int gmx_unused thread) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void spread_on_grid_lines_gpu(struct gmx_pme_t *pme, pme_atomcomm_t *atc,
         int grid_index,
         pmegrid_t *pmegrid) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void spread_on_grid_gpu(struct gmx_pme_t *pme, pme_atomcomm_t *atc,
         int grid_index,
         pmegrid_t *pmegrid) CUDA_FUNC_TERM
 // FFT

CUDA_FUNC_QUALIFIER void pme_gpu_init(gmx_pme_gpu_t **pmeGPU) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_update_flags(
        gmx_pme_gpu_t *pmeGPU,
        gmx_bool fftgridIsOnDeviceBetweenSpreadAndR2C,
        gmx_bool cfftgridIsOnDeviceBetweenR2CAndSolve,
        gmx_bool cfftgridIsOnDeviceBetweenSolveAndC2R
        ) CUDA_FUNC_TERM
//yupinov - this isn't just about grids though


CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_destroy_gpu(gmx_parallel_3dfft_gpu_t gmx_unused pfft_setup) CUDA_FUNC_TERM

inline int gmx_parallel_3dfft_real_limits_wrapper(struct gmx_pme_t *pme,
                               int                       grid_index,
                               ivec                      local_ndata,
                               ivec                      local_offset,
                               ivec                      local_size)
{
    int res = 0;
    res = gmx_parallel_3dfft_real_limits(pme->pfft_setup[grid_index], local_ndata, local_offset, local_size);
    if (pme->bGPU)
        gmx_parallel_3dfft_real_limits_gpu(pme->pfft_setup_gpu[grid_index], local_ndata, local_offset, local_size);
    return res;
}

inline int gmx_parallel_3dfft_complex_limits_wrapper(struct gmx_pme_t *pme,
                               int                       grid_index,
                               ivec                      complex_order,
                               ivec                      local_ndata,
                               ivec                      local_offset,
                               ivec                      local_size)
{
    //yupinov - so both FFT limits functiosn are broken for now? as well as constructr
    int res = 0;
    res = gmx_parallel_3dfft_complex_limits(pme->pfft_setup[grid_index], complex_order, local_ndata, local_offset, local_size);
    if (pme->bGPU)
        gmx_parallel_3dfft_complex_limits_gpu(pme->pfft_setup_gpu[grid_index], complex_order, local_ndata, local_offset, local_size);
    return res;
}

inline int gmx_parallel_3dfft_execute_wrapper(struct gmx_pme_t gmx_unused *pme,
                           int grid_index,
                           enum gmx_fft_direction gmx_unused  dir,
                           int           gmx_unused           thread,
                           gmx_wallcycle_t         wcycle)
{
    int res = 0;
    gmx_bool bGPUFFT = pme->bGPUFFT;
    int wcycle_id = ewcPME_FFT;
    int wsubcycle_id = (dir == GMX_FFT_REAL_TO_COMPLEX) ? ewcsPME_FFT_R2C : ewcsPME_FFT_C2R;  //yupinov - this is 1 thread!

    if (thread == 0)
    {
        wallcycle_start(wcycle, wcycle_id);
        wallcycle_sub_start(wcycle, wsubcycle_id);
    }

    if (bGPUFFT)
    {
        if (thread == 0)
            gmx_parallel_3dfft_execute_gpu(pme->pfft_setup_gpu[grid_index], dir, thread, wcycle, pme);
    }
    else
        res = gmx_parallel_3dfft_execute(pme->pfft_setup[grid_index], dir, thread, wcycle);

    if (thread == 0)
    {
        wallcycle_stop(wcycle, wcycle_id);
        wallcycle_sub_stop(wcycle, wsubcycle_id);
    }

    return res;
}

CUDA_FUNC_QUALIFIER void solve_pme_yzx_gpu(real gmx_unused pme_epsilon_r,
              int gmx_unused nx, int gmx_unused ny, int gmx_unused nz,
              ivec gmx_unused complex_order, ivec gmx_unused local_ndata, ivec gmx_unused local_offset, ivec gmx_unused local_size,
              splinevec gmx_unused pme_bsp_mod,
              matrix gmx_unused work_vir_q, real gmx_unused *work_energy_q,
              t_complex gmx_unused *grid,
              real gmx_unused ewaldcoeff, real gmx_unused vol,
              gmx_bool gmx_unused bEnerVir,
             gmx_pme_t *pme) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void solve_pme_lj_yzx_gpu(int gmx_unused nx, int gmx_unused ny, int gmx_unused nz,
             ivec gmx_unused complex_order, ivec gmx_unused local_ndata, ivec gmx_unused local_offset, ivec gmx_unused local_size,
             real gmx_unused rxx, real gmx_unused ryx, gmx_unused real ryy, real gmx_unused rzx, real gmx_unused rzy, real gmx_unused rzz,
             //real *mhx, real *mhy, real *mhz, real *m2, real *denom, real *tmp1, real *tmp2,
             splinevec gmx_unused pme_bsp_mod,
             matrix gmx_unused work_vir_lj, real gmx_unused *work_energy_lj,
             t_complex gmx_unused **grid, gmx_bool gmx_unused bLB,
             real gmx_unused ewaldcoeff, real gmx_unused vol,
             gmx_bool gmx_unused bEnerVir, gmx_pme_t *pme,
                                              int gmx_unused nthread, int gmx_unused thread) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gather_f_bsplines_gpu_2_pre
(gmx_bool gmx_unused bClearF,
 int gmx_unused *spline_ind, int gmx_unused spline_n,
 real gmx_unused *atc_coefficient, rvec gmx_unused *atc_f,
 real gmx_unused scale, int gmx_unused thread) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gather_f_bsplines_gpu_2
(real gmx_unused *grid, gmx_bool gmx_unused bClearF,
 int gmx_unused order,
 int gmx_unused nx, int gmx_unused ny, int gmx_unused nz, int gmx_unused pnx, int gmx_unused pny, int gmx_unused pnz,
 real gmx_unused rxx, real gmx_unused ryx, real gmx_unused ryy, real gmx_unused rzx, real gmx_unused rzy, real gmx_unused rzz,
 int gmx_unused *spline_ind, int gmx_unused spline_n,
 real gmx_unused *atc_coefficient, rvec gmx_unused *atc_f, ivec gmx_unused *atc_idx,
 splinevec gmx_unused *spline_theta, splinevec gmx_unused *spline_dtheta,
 real gmx_unused scale, gmx_pme_t *pme, int gmx_unused thread
 ) CUDA_FUNC_TERM



#endif // PMEGPU_H
