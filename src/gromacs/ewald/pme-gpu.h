/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016, by the GROMACS development team, led by
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

/*! \libinternal \file
 *
 * \brief This file contains function definitions for performing the PME calculations on GPU.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \ingroup module_ewald
 */

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
                                                     ivec                      CUDA_FUNC_ARGUMENT(ndata),
                                                     gmx_pme_t                *CUDA_FUNC_ARGUMENT(pme))  CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_real_limits_gpu(
        gmx_parallel_3dfft_gpu_t CUDA_FUNC_ARGUMENT(pfft_setup),
        ivec                     CUDA_FUNC_ARGUMENT(local_ndata),
        ivec                     CUDA_FUNC_ARGUMENT(local_offset),
        ivec                     CUDA_FUNC_ARGUMENT(local_size)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_complex_limits_gpu(gmx_parallel_3dfft_gpu_t CUDA_FUNC_ARGUMENT(pfft_setup),
                                                               ivec                     CUDA_FUNC_ARGUMENT(local_ndata),
                                                               ivec                     CUDA_FUNC_ARGUMENT(local_offset),
                                                               ivec                     CUDA_FUNC_ARGUMENT(local_size)) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_execute_gpu(gmx_pme_t             *CUDA_FUNC_ARGUMENT(pme),
                                                        enum gmx_fft_direction CUDA_FUNC_ARGUMENT(dir),
                                                        const int              CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void spread_on_grid_gpu(gmx_pme_t      *CUDA_FUNC_ARGUMENT(pme),
                                            pme_atomcomm_t *CUDA_FUNC_ARGUMENT(atc),
                                            const int       CUDA_FUNC_ARGUMENT(grid_index),
                                            pmegrid_t      *CUDA_FUNC_ARGUMENT(pmegrid),
                                            const gmx_bool  CUDA_FUNC_ARGUMENT(bCalcSplines),
                                            const gmx_bool  CUDA_FUNC_ARGUMENT(bSpread),
                                            const gmx_bool  CUDA_FUNC_ARGUMENT(bDoSplines)
                                            ) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gather_f_bsplines_gpu(gmx_pme_t      *CUDA_FUNC_ARGUMENT(pme),
                                               const gmx_bool  CUDA_FUNC_ARGUMENT(bOverwriteForces)) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void solve_pme_gpu(
        gmx_pme_t     *CUDA_FUNC_ARGUMENT(pme),
        t_complex     *CUDA_FUNC_ARGUMENT(grid),
        const gmx_bool CUDA_FUNC_ARGUMENT(bEnerVir)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void pme_gpu_get_forces(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM
CUDA_FUNC_QUALIFIER void pme_gpu_get_energy_virial(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

// these should not really be external - only used in GPU launch code which is stuck in pme.cpp
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_gather(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM_WITH_RETURN(FALSE)
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_FFT(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM_WITH_RETURN(FALSE)
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_wrapping(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM_WITH_RETURN(FALSE)
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_solve(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM_WITH_RETURN(FALSE)
CUDA_FUNC_QUALIFIER void pme_gpu_sync_grid(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), enum gmx_fft_direction CUDA_FUNC_ARGUMENT(dir)) CUDA_FUNC_TERM

// nice external functions

/*! \brief Finds out if PME is ran on GPU currently. */
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_enabled(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM_WITH_RETURN(FALSE)

/*! \brief Initializes the PME GPU data at the beginning or on DD. */
CUDA_FUNC_QUALIFIER void pme_gpu_init(gmx_pme_gpu_t      **CUDA_FUNC_ARGUMENT(pmeGPU),
                                      gmx_pme_t           *CUDA_FUNC_ARGUMENT(pme),
                                      const gmx_hw_info_t *CUDA_FUNC_ARGUMENT(hwinfo),
                                      const gmx_gpu_opt_t *CUDA_FUNC_ARGUMENT(gpu_opt)) CUDA_FUNC_TERM

/*! \brief Destroys the PME GPU data at the end of the run. */
CUDA_FUNC_QUALIFIER void pme_gpu_deinit(gmx_pme_t **CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

/*! \brief Initializes the PME GPU step. */
CUDA_FUNC_QUALIFIER void pme_gpu_step_init(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

/*! \brief Sets the PME GPU constants. Currently it is called together with the pme_gpu_step_init, but can possibly be decoupled? */
CUDA_FUNC_QUALIFIER void pme_gpu_set_constants(gmx_pme_t   *CUDA_FUNC_ARGUMENT(pme),
                                               const matrix CUDA_FUNC_ARGUMENT(box),
                                               const real   CUDA_FUNC_ARGUMENT(ewaldCoeff)) CUDA_FUNC_TERM

/*! \brief Initializes the single grid in the PME GPU step. */
CUDA_FUNC_QUALIFIER void pme_gpu_grid_init(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), const int CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM



/*! \brief Finishes the PME GPU step, copying back the forces and/or energy/virial. */
CUDA_FUNC_QUALIFIER void pme_gpu_step_end(gmx_pme_t     *CUDA_FUNC_ARGUMENT(pme),
                                          const gmx_bool CUDA_FUNC_ARGUMENT(bCalcF),
                                          const gmx_bool CUDA_FUNC_ARGUMENT(bCalcEnerVir)) CUDA_FUNC_TERM

/*! \brief Resets the PME GPU timings. */
CUDA_FUNC_QUALIFIER void pme_gpu_reset_timings(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void pme_gpu_get_timings(gmx_wallclock_gpu_t **CUDA_FUNC_ARGUMENT(timings), gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

#endif // PMEGPU_H
