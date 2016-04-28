#ifndef PMEGPU_H
#define PMEGPU_H

#include "pme-internal.h"
#include "gromacs/gpu_utils/gpu_macros.h"


//yupinov add pmegpu detection warning!
//yupinov add author info everywhere
//yupinov CUDA_FUNC_QUALIFIER everywhere? as well as parameters
// gmx_unused


/*! \brief Resets PME GPU timings. */
CUDA_FUNC_QUALIFIER void pme_gpu_reset_timings(struct gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

// internal data handling

// copies the grid sizes for overlapping (used in PME wrap/unwrap)
CUDA_FUNC_QUALIFIER void pme_gpu_copy_wrap_zones(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

// copies the reciprocal box to the device (used in PME spread/solve/gather)
CUDA_FUNC_QUALIFIER void pme_gpu_copy_recipbox(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

// copies the bspline moduli to the device (used in PME solve)
CUDA_FUNC_QUALIFIER void pme_gpu_copy_bspline_moduli(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

// copies the coordinates to the device (used in PME spread)
CUDA_FUNC_QUALIFIER void pme_gpu_copy_coordinates(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM
// copies the charges to the device (used in PME spread/gather)
CUDA_FUNC_QUALIFIER void pme_gpu_copy_charges(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM



// copies the nn and fsh to the device (used in PME spread(spline))
CUDA_FUNC_QUALIFIER void pme_gpu_copy_calcspline_constants(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

// clearing
CUDA_FUNC_QUALIFIER void pme_gpu_clear_grid(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), const int CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_clear_energy_virial(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), const int CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM

// allocating
CUDA_FUNC_QUALIFIER void pme_gpu_alloc_grid(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), const int CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_alloc_energy_virial(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), const int CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_alloc_gather_forces(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_init_gpu(gmx_parallel_3dfft_gpu_t *CUDA_FUNC_ARGUMENT(pfft_setup),
                               ivec CUDA_FUNC_ARGUMENT(ndata),
                               gmx_pme_t *CUDA_FUNC_ARGUMENT(pme))  CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_real_limits_gpu(gmx_parallel_3dfft_gpu_t gmx_unused pfft_setup,
                               ivec          gmx_unused            local_ndata,
                               ivec         gmx_unused             local_offset,
                               ivec         gmx_unused             local_size) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_complex_limits_gpu(gmx_parallel_3dfft_gpu_t gmx_unused pfft_setup,
                                  ivec     gmx_unused                 complex_order,
                                  ivec       gmx_unused               local_ndata,
                                  ivec       gmx_unused               local_offset,
                                  ivec       gmx_unused               local_size) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_execute_gpu(const gmx_parallel_3dfft_gpu_t &CUDA_FUNC_ARGUMENT(pfft_setup),
                           enum gmx_fft_direction CUDA_FUNC_ARGUMENT(dir),
                            gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void spread_on_grid_gpu(struct gmx_pme_t *pme, pme_atomcomm_t *atc,
        int grid_index,
        pmegrid_t *pmegrid,
        const gmx_bool bCalcSplines,
        const gmx_bool bSpread,
        const gmx_bool bDoSplines
) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gather_f_bsplines_gpu(struct gmx_pme_t *pme, real *grid,
              pme_atomcomm_t *atc,
              splinedata_t *spline,
              real scale) CUDA_FUNC_TERM



 // FFT

struct gmx_hw_info_t;
struct gmx_gpu_opt_t;


// nice external functions

CUDA_FUNC_QUALIFIER void pme_gpu_init(gmx_pme_gpu_t **CUDA_FUNC_ARGUMENT(pmeGPU),
                                      gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                      const gmx_hw_info_t *CUDA_FUNC_ARGUMENT(hwinfo),
                                      const gmx_gpu_opt_t *CUDA_FUNC_ARGUMENT(gpu_opt)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_deinit(//gmx_pme_gpu_t **CUDA_FUNC_ARGUMENT(pmeGPU),
                                      gmx_pme_t **CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void pme_gpu_step_init(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_step_end(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                          const gmx_bool CUDA_FUNC_ARGUMENT(bCalcF),
                                          const gmx_bool CUDA_FUNC_ARGUMENT(bCalcEnerVir)) CUDA_FUNC_TERM

// every grid has different coefficients, etc. - several grids per step;
//CUDA_FUNC_QUALIFIER void pme_gpu_grid_init(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM



CUDA_FUNC_QUALIFIER void pme_gpu_update_flags(
        gmx_pme_gpu_t *pmeGPU,
        gmx_bool keepGPUDataBetweenSpreadAndR2C,
        gmx_bool keepGPUDataBetweenR2CAndSolve,
        gmx_bool keepGPUDataBetweenSolveAndC2R,
        gmx_bool keepGPUDataBetweenC2RAndGather
        ) CUDA_FUNC_TERM //?
//yupinov - this isn't just about grids though


CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_destroy_gpu(const gmx_parallel_3dfft_gpu_t &CUDA_FUNC_ARGUMENT(pfft_setup)) CUDA_FUNC_TERM

inline int gmx_parallel_3dfft_real_limits_wrapper(struct gmx_pme_t *pme,
                               int                       grid_index,
                               ivec                      local_ndata,
                               ivec                      local_offset,
                               ivec                      local_size)
{
    int res = 0;
    res = gmx_parallel_3dfft_real_limits(pme->pfft_setup[grid_index], local_ndata, local_offset, local_size);
    if (pme->bGPUFFT)
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
    if (pme->bGPUFFT)
        gmx_parallel_3dfft_complex_limits_gpu(pme->pfft_setup_gpu[grid_index], complex_order, local_ndata, local_offset, local_size);

    return res;
}

#include "gromacs/utility/gmxomp.h"

int gmx_parallel_3dfft_execute_wrapper(struct gmx_pme_t gmx_unused *pme,
                           int grid_index,
                           enum gmx_fft_direction gmx_unused  dir,
                           gmx_wallcycle_t         wcycle);

int solve_pme_yzx_wrapper(struct gmx_pme_t *pme, t_complex *grid,
                  real ewaldcoeff, real vol,
                  gmx_bool bEnerVir);

CUDA_FUNC_QUALIFIER void solve_pme_gpu
(gmx_pme_t *pme, t_complex *grid,
                  real ewaldcoeff, real vol,
                  gmx_bool bEnerVir) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER int solve_pme_lj_yzx_gpu(int gmx_unused nx, int gmx_unused ny, int gmx_unused nz,
             ivec gmx_unused complex_order, ivec gmx_unused local_ndata, ivec gmx_unused local_offset, ivec gmx_unused local_size,
             real gmx_unused rxx, real gmx_unused ryx, gmx_unused real ryy, real gmx_unused rzx, real gmx_unused rzy, real gmx_unused rzz,
             //real *mhx, real *mhy, real *mhz, real *m2, real *denom, real *tmp1, real *tmp2,
             splinevec gmx_unused pme_bsp_mod,
             matrix gmx_unused work_vir_lj, real gmx_unused *work_energy_lj,
             t_complex gmx_unused **grid, gmx_bool gmx_unused bLB,
             real gmx_unused ewaldcoeff, real gmx_unused vol,
             gmx_bool gmx_unused bEnerVir, gmx_pme_t *pme,
                                              int gmx_unused nthread, int gmx_unused thread) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void pme_gpu_get_forces(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_get_energy_virial(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

#endif // PMEGPU_H
