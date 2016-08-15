#ifndef PMEGPU_H
#define PMEGPU_H

#include "gromacs/gpu_utils/gpu_macros.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/math/gmxcomplex.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/timing/gpu_timing.h"

//yupinov add author info everywhere
// gmx_unused

#include "pme-internal.h"

struct gmx_hw_info_t;
struct gmx_gpu_opt_t;

// internal data handling

// copies the bspline moduli to the device (used in PME solve)
CUDA_FUNC_QUALIFIER void pme_gpu_copy_bspline_moduli(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

// copies the charges to the device (used in PME spread/gather)
CUDA_FUNC_QUALIFIER void pme_gpu_copy_charges(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM



// copies the nn and fsh to the device (used in PME spread(spline))
CUDA_FUNC_QUALIFIER void pme_gpu_copy_calcspline_constants(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

// clearing
CUDA_FUNC_QUALIFIER void pme_gpu_clear_grid(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), const int CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_clear_energy_virial(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), const int CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM

// allocating
CUDA_FUNC_QUALIFIER void pme_gpu_alloc_grids(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), const int CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_alloc_energy_virial(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), const int CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_alloc_gather_forces(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_init_gpu(gmx_parallel_3dfft_gpu_t *CUDA_FUNC_ARGUMENT(pfft_setup),
                               ivec CUDA_FUNC_ARGUMENT(ndata),
                               gmx_pme_t *CUDA_FUNC_ARGUMENT(pme))  CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_real_limits_gpu(
                               gmx_parallel_3dfft_gpu_t CUDA_FUNC_ARGUMENT(pfft_setup),
                               ivec CUDA_FUNC_ARGUMENT(local_ndata),
                               ivec CUDA_FUNC_ARGUMENT(local_offset),
                               ivec CUDA_FUNC_ARGUMENT(local_size)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_complex_limits_gpu(gmx_parallel_3dfft_gpu_t CUDA_FUNC_ARGUMENT(pfft_setup),
                                  ivec CUDA_FUNC_ARGUMENT(local_ndata),
                                  ivec CUDA_FUNC_ARGUMENT(local_offset),
                                  ivec CUDA_FUNC_ARGUMENT(local_size)) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_execute_gpu(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                           enum gmx_fft_direction CUDA_FUNC_ARGUMENT(dir),
                           const int CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void spread_on_grid_gpu(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                            pme_atomcomm_t *CUDA_FUNC_ARGUMENT(atc),
        const int CUDA_FUNC_ARGUMENT(grid_index),
        pmegrid_t *CUDA_FUNC_ARGUMENT(pmegrid),
        const gmx_bool CUDA_FUNC_ARGUMENT(bCalcSplines),
        const gmx_bool CUDA_FUNC_ARGUMENT(bSpread),
        const gmx_bool CUDA_FUNC_ARGUMENT(bDoSplines)
) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gather_f_bsplines_gpu(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
              real *CUDA_FUNC_ARGUMENT(grid),
              pme_atomcomm_t *CUDA_FUNC_ARGUMENT(atc),
              const real CUDA_FUNC_ARGUMENT(scale),
              const gmx_bool CUDA_FUNC_ARGUMENT(bOverwriteForces)) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void solve_pme_gpu(
                  gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                  t_complex *CUDA_FUNC_ARGUMENT(grid),
                  const gmx_bool CUDA_FUNC_ARGUMENT(bEnerVir)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void pme_gpu_get_forces(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_get_energy_virial(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

// these should not really be external - only used in GPU launch code which is stuck in pme.cpp
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_gather(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_FFT(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_wrapping(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_solve(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_sync_grid(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), enum gmx_fft_direction CUDA_FUNC_ARGUMENT(dir)) CUDA_FUNC_TERM

// nice external functions

/*! \brief Resets PME GPU timings. */
CUDA_FUNC_QUALIFIER void pme_gpu_reset_timings(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

/*! \brief Initializes the PME GPU data at the beginning or on DD. */
CUDA_FUNC_QUALIFIER void pme_gpu_init(gmx_pme_gpu_t **CUDA_FUNC_ARGUMENT(pmeGPU),
                                      gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                      const gmx_hw_info_t *CUDA_FUNC_ARGUMENT(hwinfo),
                                      const gmx_gpu_opt_t *CUDA_FUNC_ARGUMENT(gpu_opt)) CUDA_FUNC_TERM

/*! \brief Destroys the PME GPU data at the end. */
CUDA_FUNC_QUALIFIER void pme_gpu_deinit(//gmx_pme_gpu_t **CUDA_FUNC_ARGUMENT(pmeGPU),
                                      gmx_pme_t **CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

/*! \brief Initializes the PME GPU step. */
CUDA_FUNC_QUALIFIER void pme_gpu_step_init(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

/*! \brief Sets the PME GPU constants. Is there any reason for this to be separate from the pme_gpu_step_init? */
CUDA_FUNC_QUALIFIER void pme_gpu_set_constants(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                               const matrix CUDA_FUNC_ARGUMENT(box),
                                               const real CUDA_FUNC_ARGUMENT(ewaldCoeff)) CUDA_FUNC_TERM


/*! \brief Finishes the PME GPU step, copying back the forces and/or energy/virial. */
CUDA_FUNC_QUALIFIER void pme_gpu_step_end(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                          const gmx_bool CUDA_FUNC_ARGUMENT(bCalcF),
                                          const gmx_bool CUDA_FUNC_ARGUMENT(bCalcEnerVir)) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_destroy_gpu(const gmx_parallel_3dfft_gpu_t &CUDA_FUNC_ARGUMENT(pfft_setup)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void pme_gpu_get_timings(gmx_wallclock_gpu_t **CUDA_FUNC_ARGUMENT(timings), gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

#endif // PMEGPU_H
