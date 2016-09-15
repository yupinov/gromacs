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
#include "pme-internal.h"

struct gmx_hw_info_t;
struct gmx_gpu_opt_t;


//yupinov add author info everywhere


// internal data handling


CUDA_FUNC_QUALIFIER void gmx_parallel_3dfft_execute_gpu(gmx_pme_t             *CUDA_FUNC_ARGUMENT(pme),
                                                        enum gmx_fft_direction CUDA_FUNC_ARGUMENT(dir),
                                                        const int              CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM


CUDA_FUNC_QUALIFIER void spread_on_grid_gpu(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                            pme_atomcomm_t  *CUDA_FUNC_ARGUMENT(atc),
                                            const int        CUDA_FUNC_ARGUMENT(grid_index),
                                            pmegrid_t       *CUDA_FUNC_ARGUMENT(pmegrid),
                                            const gmx_bool   CUDA_FUNC_ARGUMENT(bCalcSplines),
                                            const gmx_bool   CUDA_FUNC_ARGUMENT(bSpread),
                                            const gmx_bool   CUDA_FUNC_ARGUMENT(bDoSplines)
                                            ) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void gather_f_bsplines_gpu(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                               const gmx_bool   CUDA_FUNC_ARGUMENT(bOverwriteForces)) CUDA_FUNC_TERM

CUDA_FUNC_QUALIFIER void solve_pme_gpu(
        gmx_pme_t     *CUDA_FUNC_ARGUMENT(pme),
        t_complex     *CUDA_FUNC_ARGUMENT(grid),
        const gmx_bool CUDA_FUNC_ARGUMENT(bEnerVir)) CUDA_FUNC_TERM

// these should not really be external - only used in GPU launch code which is stuck in pme.cpp
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_gather(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM_WITH_RETURN(FALSE)
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_FFT(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM_WITH_RETURN(FALSE)
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_wrapping(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM_WITH_RETURN(FALSE)
CUDA_FUNC_QUALIFIER gmx_bool pme_gpu_performs_solve(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM_WITH_RETURN(FALSE)
CUDA_FUNC_QUALIFIER void pme_gpu_sync_grid(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme), const gmx_fft_direction CUDA_FUNC_ARGUMENT(dir)) CUDA_FUNC_TERM

// nice external functions

/*! \brief
 * Finds out if PME is set to run on GPU.
 *
 * \param[in] pme  The PME structure.
 * \returns        TRUE if PME runs on GPU, FALSE otherwise.
 */
gmx_inline gmx_bool pme_gpu_enabled(const gmx_pme_t *pme)
{
    /* Something to think about: should this function be called from all the CUDA_FUNC_QUALIFIER functions?
     * In other words, should we plan for dynamic toggling of the PME GPU?
     */
    return (pme != NULL) && pme->bGPU;
}

/*! \brief
 * Initializes the PME GPU data at the beginning of the run or on DLB. Does nothing on non-CUDA builds.
 *
 * \param[in] pme     The PME structure.
 * \param[in] hwinfo  The hardware information structure.
 * \param[in] gpu_opt The GPU information structure.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_init(gmx_pme_t           *CUDA_FUNC_ARGUMENT(pme),
                                      const gmx_hw_info_t *CUDA_FUNC_ARGUMENT(hwinfo),
                                      const gmx_gpu_opt_t *CUDA_FUNC_ARGUMENT(gpu_opt)) CUDA_FUNC_TERM

/*! \brief
 * Destroys the PME GPU data at the end of the run. Does nothing on non-CUDA builds.
 *
 * \param[in] pme     The PME structure.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_deinit(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

/*! \brief
 * Initializes the PME GPU step (copies coordinates onto GPU, possibly sets the unit cell parameters). Does nothing on non-CUDA builds.
 *
 * \param[in] pme     The PME structure.
 * \param[in] box     The unit cell box which does not necessarily change every step (only with pressure coupling enabled).
 *                    Currently it is simply compared with the previous one to determine if it needs to be updated.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_step_init(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                           const matrix     CUDA_FUNC_ARGUMENT(box)) CUDA_FUNC_TERM

/*! \brief
 * Sets the PME GPU constants. Does nothing on non-CUDA builds.
 *
 * \param[in] pme            The PME structure.
 * \param[in] ewaldCoeff     The Ewald coefficient.
 *
 * Currently it is called together with the pme_gpu_step_init, but should be called in pme_gpu_init (on grid size change) instead.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_set_constants(const gmx_pme_t    *CUDA_FUNC_ARGUMENT(pme),
                                               const float         CUDA_FUNC_ARGUMENT(ewaldCoeff)) CUDA_FUNC_TERM

/*! \brief
 * Reallocates the local atoms data (charges, coordinates, etc.). Copies the charges. Does nothing on non-CUDA builds.
 *
 * \param[in] pme            The PME structure.
 * \param[in] nAtoms         The number of particles.
 * \param[in] coefficients   The pointer to the host-side array of particle charges.
 *
 * This is a function that should only be called in the beginning of the run and on domain decomposition.
 * Should be called before the pme_gpu_set_io_ranges.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_reinit_atoms(const gmx_pme_t  *CUDA_FUNC_ARGUMENT(pme),
                                              const int         CUDA_FUNC_ARGUMENT(nAtoms),
                                              float            *CUDA_FUNC_ARGUMENT(coefficients)) CUDA_FUNC_TERM


/*! \brief
 * Allocates the local atoms data (charges, coordinates, etc.) at the very first MD step. Copies the charges. Does nothing on non-CUDA builds.
 *
 * \param[in] pme            The PME structure.
 * \param[in] nAtoms         The number of particles.
 * \param[in] coefficients   The pointer to the host-side array of particle charges.
 *
 * This is a wrapper just for calling pme_gpu_reinit_atoms once at the beginning of the run.
 * There is probably more elegant way to do this...
 */
CUDA_FUNC_QUALIFIER void pme_gpu_init_atoms_once(const gmx_pme_t  *CUDA_FUNC_ARGUMENT(pme),
                                                 const int         CUDA_FUNC_ARGUMENT(nAtoms),
                                                 float            *CUDA_FUNC_ARGUMENT(coefficients)) CUDA_FUNC_TERM

/*! \brief
 * Sets the host-side I/O buffers in the PME GPU. Does nothing on non-CUDA builds.
 *
 * \param[in] pme            The PME structure.
 * \param[in] coordinates    The pointer to the host-side array of particle coordinates in rvec format.
 * \param[in] forces         The pointer to the host-side array of particle forces.
 *                           It will be used for output, but can also be used for input,
 *                           if bClearForces is passed as false to the gmx_pme_gpu_launch_gather.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_set_io_ranges(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                               rvec            *CUDA_FUNC_ARGUMENT(coordinates),
                                               rvec            *CUDA_FUNC_ARGUMENT(forces)) CUDA_FUNC_TERM

/*! \brief
 * Finishes the PME GPU step, copying back the forces and/or energy/virial. Does nothing on non-CUDA builds.
 *
 * \param[in] pme            The PME structure.
 * \param[in] bCalcForces    The left-over flag from the CPU code which tells the function to copy the forces to the CPU side. Should be passed to the launch call instead.
 * \param[in] bCalcEnerVir   The left-over flag from the CPU code which tells the function to copy the energy/virial to the CPU side. Should be passed to the launch call instead.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_step_end(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                          const gmx_bool   CUDA_FUNC_ARGUMENT(bCalcForces),
                                          const gmx_bool   CUDA_FUNC_ARGUMENT(bCalcEnerVir)) CUDA_FUNC_TERM

/*! \brief
 * Resets the PME GPU timings. To be called at the reset step. Does nothing on non-CUDA builds.
 *
 * \param[in] pme            The PME structure.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_reset_timings(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

/*! \brief
 * Copies the PME GPU timings to the gmx_wallclock_gpu_t structure (for log output). To be called at the run end. Does nothing on non-CUDA builds.
 *
 * \param[in] timings           The gmx_wallclock_gpu_t structure (with some shamelessly duplicated fields for the PME GPU timings).
 * \param[in] pme               The PME structure (why is it a second argument?).
 */
CUDA_FUNC_QUALIFIER void pme_gpu_get_timings(gmx_wallclock_gpu_t **CUDA_FUNC_ARGUMENT(timings),
                                             const gmx_pme_t      *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

/*! \brief
 * Gets the PME GPU output virial/energy. Should be called after pme_gpu_step_end. Does nothing on non-CUDA builds.
 *
 * \param[in]  pme  The PME structure.
 * \param[out] energy  The output energy pointer.
 * \param[out] virial  The output virial matrix.
 */
void pme_gpu_get_energy_virial(const gmx_pme_t *pme, real *energy, matrix virial);

#endif // PMEGPU_H
