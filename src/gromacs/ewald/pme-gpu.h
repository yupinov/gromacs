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
#include "pme-gpu-types.h"
#include "pme-internal.h"

struct gmx_hw_info_t;
struct gmx_gpu_opt_t;

// internal data handling

// A GPU counterpart to gmx_parallel_3dfft_execute
CUDA_FUNC_QUALIFIER void pme_gpu_3dfft(gmx_pme_t             *CUDA_FUNC_ARGUMENT(pme),
                                       enum gmx_fft_direction CUDA_FUNC_ARGUMENT(dir),
                                       const int              CUDA_FUNC_ARGUMENT(grid_index)) CUDA_FUNC_TERM


// A GPU counterpart to the spread_on_grid
CUDA_FUNC_QUALIFIER void pme_gpu_spread(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                        pme_atomcomm_t  *CUDA_FUNC_ARGUMENT(atc),
                                        const int        CUDA_FUNC_ARGUMENT(grid_index),
                                        pmegrid_t       *CUDA_FUNC_ARGUMENT(pmegrid),
                                        const gmx_bool   CUDA_FUNC_ARGUMENT(bCalcSplines),
                                        const gmx_bool   CUDA_FUNC_ARGUMENT(bSpread)) CUDA_FUNC_TERM

// A GPU counterpart to the gather_f_bsplines
CUDA_FUNC_QUALIFIER void pme_gpu_gather(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                        const gmx_bool   CUDA_FUNC_ARGUMENT(bOverwriteForces)) CUDA_FUNC_TERM

// A GPU counterpart to the solve_pme_yzx
CUDA_FUNC_QUALIFIER void pme_gpu_solve(
        gmx_pme_t     *CUDA_FUNC_ARGUMENT(pme),
        t_complex     *CUDA_FUNC_ARGUMENT(grid),
        const gmx_bool CUDA_FUNC_ARGUMENT(bEnerVir)) CUDA_FUNC_TERM

// nice external functions

/*! \brief \internal
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

/*! \brief \internal
 * (Re-)initializes the PME GPU data at the beginning of the run or on DLB. Does nothing on non-CUDA builds.
 *
 * \param[in] pme     The PME structure.
 * \param[in] hwinfo  The hardware information structure.
 * \param[in] gpu_opt The GPU information structure.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_reinit(gmx_pme_t           *CUDA_FUNC_ARGUMENT(pme),
                                        const gmx_hw_info_t *CUDA_FUNC_ARGUMENT(hwinfo),
                                        const gmx_gpu_opt_t *CUDA_FUNC_ARGUMENT(gpu_opt)) CUDA_FUNC_TERM

/*! \brief \internal
 * Destroys the PME GPU data at the end of the run. Does nothing on non-CUDA builds.
 *
 * \param[in] pme     The PME structure.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_destroy(gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

/*! \brief
 * Starts the PME GPU step (copies coordinates onto GPU, possibly sets the unit cell parameters). Does nothing on non-CUDA builds.
 *
 * \param[in] pme     The PME structure.
 * \param[in] box     The unit cell box which does not necessarily change every step (only with pressure coupling enabled).
 *                    Currently it is simply compared with the previous one to determine if it needs to be updated.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_start_step(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                            const matrix     CUDA_FUNC_ARGUMENT(box)) CUDA_FUNC_TERM

/*! \brief \internal
 * Finishes the PME GPU step, waiting for the output forces and/or energy/virial to be copied to the host. Does nothing on non-CUDA builds.
 *
 * \param[in] pme            The PME structure.
 * \param[in] bCalcForces    The left-over flag from the CPU code which tells the function to copy the forces to the CPU side. Should be passed to the launch call instead.
 * \param[in] bCalcEnerVir   The left-over flag from the CPU code which tells the function to copy the energy/virial to the CPU side. Should be passed to the launch call instead.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_finish_step(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                             const gmx_bool   CUDA_FUNC_ARGUMENT(bCalcForces),
                                             const gmx_bool   CUDA_FUNC_ARGUMENT(bCalcEnerVir)) CUDA_FUNC_TERM

/*! \brief \internal
 * Gets the PME GPU output virial/energy. Should be called after pme_gpu_finish_step. Does nothing on non-CUDA builds.
 *
 * \param[in]  pme  The PME structure.
 * \param[out] energy  The output energy pointer.
 * \param[out] virial  The output virial matrix.
 *
 * Should thsi be merged with pme_gpu_finish_step?
 */
void pme_gpu_get_energy_virial(const gmx_pme_t *pme, real *energy, matrix virial);

/*! \brief \internal
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

/*! \brief \internal
 * Sets the host-side I/O buffers in the PME GPU. Does nothing on non-CUDA builds.
 *
 * \param[in] pme            The PME structure.
 * \param[in] coordinates    The pointer to the host-side array of particle coordinates in rvec format.
 * \param[in] forces         The pointer to the host-side array of particle forces.
 *                           It will be used for output, but can also be used for input,
 *                           if bClearForces is passed as false to the pme_gpu_launch_gather.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_set_io_ranges(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme),
                                               rvec            *CUDA_FUNC_ARGUMENT(coordinates),
                                               rvec            *CUDA_FUNC_ARGUMENT(forces)) CUDA_FUNC_TERM


/*! \brief \internal
 * Resets the PME GPU timings. To be called at the reset step. Does nothing on non-CUDA builds.
 *
 * \param[in] pme            The PME structure.
 */
CUDA_FUNC_QUALIFIER void pme_gpu_reset_timings(const gmx_pme_t *CUDA_FUNC_ARGUMENT(pme)) CUDA_FUNC_TERM

/*! \brief \internal
 * Copies the PME GPU timings to the gmx_wallclock_gpu_t structure (for log output). To be called at the run end. Does nothing on non-CUDA builds.
 *
 * \param[in] pme               The PME structure
 * \param[in] timings           The gmx_wallclock_gpu_t structure (with some shamelessly duplicated fields for the PME GPU timings).
 */
CUDA_FUNC_QUALIFIER void pme_gpu_get_timings(const gmx_pme_t      *CUDA_FUNC_ARGUMENT(pme),
                                             gmx_wallclock_gpu_t **CUDA_FUNC_ARGUMENT(timings)) CUDA_FUNC_TERM




/*! \libinternal
 * \brief
 *
 * Tells if PME performs the gathering stage on GPU.
 *
 * \param[in] pme            The PME data structure.
 * \returns                  TRUE if the gathering is performed on GPU, FALSE otherwise.
 */
gmx_inline gmx_bool pme_gpu_performs_gather(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && pme->gpu->bGPUGather;
}

/*! \libinternal
 * \brief
 *
 * Tells if PME performs the FFT stages on GPU.
 *
 * \param[in] pme            The PME data structure.
 * \returns                  TRUE if FFT is performed on GPU, FALSE otherwise.
 */
gmx_inline gmx_bool pme_gpu_performs_FFT(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && pme->gpu->bGPUFFT;
}

/*! \libinternal
 * \brief
 *
 * Tells if PME performs the grid (un-)wrapping on GPU.
 *
 * \param[in] pme            The PME data structure.
 * \returns                  TRUE if (un-)wrapping is performed on GPU, FALSE otherwise.
 */
gmx_inline gmx_bool pme_gpu_performs_wrapping(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && pme->gpu->bGPUSingle;
}

/*! \brief \internal
 * Tells if PME performs the grid solving on GPU.
 *
 * \param[in] pme            The PME data structure.
 * \returns                  TRUE if solving is performed on GPU, FALSE otherwise.
 */
gmx_inline gmx_bool pme_gpu_performs_solve(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && pme->gpu->bGPUSolve;
}

/*! \brief \internal
 * Tells if PME runs on multiple GPUs.
 *
 * \param[in] pme            The PME data structure.
 * \returns                  TRUE if PME runs on multiple GPUs, FALSE otherwise.
 */
gmx_inline gmx_bool pme_gpu_uses_dd(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && !pme->gpu->bGPUSingle;
}




void pme_gpu_get_results(const gmx_pme_t *pme,
                         gmx_wallcycle_t  wcycle,
                         matrix           vir_q,
                         real            *energy_q,
                         int              flags);

// launches first part of PME GPU - from spread up to and including FFT C2R
// and copying energy/virial back
CUDA_FUNC_QUALIFIER void pme_gpu_launch(gmx_pme_t         *CUDA_FUNC_ARGUMENT(pme),
                                        int                CUDA_FUNC_ARGUMENT(nAtoms),
                                        rvec              *CUDA_FUNC_ARGUMENT(x),
                                        rvec              *CUDA_FUNC_ARGUMENT(f),
                                        real              *CUDA_FUNC_ARGUMENT(charges),
                                        matrix             CUDA_FUNC_ARGUMENT(box),
                                        gmx_wallcycle_t    CUDA_FUNC_ARGUMENT(wcycle),
                                        int                CUDA_FUNC_ARGUMENT(flags)) CUDA_FUNC_TERM


// launches the rest of the PME GPU:
// copying calculated forces (e.g. listed) onto GPU (only for bClearF == false), gather, copying forces back
// for separate PME ranks there is no precalculated forces, so bClearF has to be true
// so there is no reason not to put this call directly back into pme_gpu_launch for bClearF == true
void pme_gpu_launch_gather(const gmx_pme_t      *pme,
                           gmx_wallcycle_t       wcycle,
                           gmx_bool              bClearForces);





#endif // PMEGPU_H
