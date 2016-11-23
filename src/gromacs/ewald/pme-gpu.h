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
 * \brief This file contains exposed function definitions for performing the PME calculations on GPU.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \inlibraryapi
 * \ingroup module_ewald
 */

#ifndef PMEGPU_H
#define PMEGPU_H

#include "gromacs/fft/fft.h"
#include "gromacs/timing/wallcycle.h"

struct gmx_pme_t;
struct gmx_wallclock_gpu_pme_t;
struct pme_gpu_t;

/*! \brief
 * Tells if PME is enabled to run on GPU (not necessarily active at the moment).
 * For now, this decision is stored in the PME structure itself.
 * FIXME: this is an information that should be managed by the task scheduler.
 * As soon as such functionality appears, this function should be removed from this module.
 *
 * \param[in]  pme            The PME data structure.
 * \returns true if PME can run on GPU, false otherwise.
 */
bool pme_gpu_task_enabled(const gmx_pme_t *pme);

/*! \brief
 * Resets the PME GPU timings. To be called at the reset step.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_reset_timings(const gmx_pme_t *pme);

/*! \brief
 * Copies the PME GPU timings to the gmx_wallclock_gpu_t structure (for log output). To be called at the run end.
 *
 * \param[in] pme               The PME structure.
 * \param[in] timings           The gmx_wallclock_gpu_t structure.
 */
void pme_gpu_get_timings(const gmx_pme_t         *pme,
                         gmx_wallclock_gpu_pme_t *timings);

/* The main PME GPU functions */

/*! \brief \internal
 * Launches most of the PME GPU stages, except for gathering and its force manipulations.
 * The FFT can be launched on CPU instead of GPU.
 *
 * \param[in]  pme            The PME data structure.
 * \param[in]  nAtoms         The number of local atoms.
 *                            Only needed for the first MD step (late initialization).
 * \param[in]  x              The array of local atoms' coordinates.
 * \param[in]  charges        The array of local atoms' charges.
 *                            Only needed for the first MD step (late initialization).
 * \param[in]  box            The unit cell box.
 * \param[in]  wcycle         The wallclock counter.
 * \param[in]  flags          The combination of flags to affect the PME computation.
 *                            TODO: document, rethink the flag handling.
 */
void pme_gpu_launch_everything_but_gather(gmx_pme_t               *pme,
                                          int                      nAtoms,
                                          const rvec              *x,
                                          const  real             *charges,
                                          const matrix             box,
                                          gmx_wallcycle_t          wcycle,
                                          int                      flags);

/*! \brief \internal
 * Launches the PME GPU gathering and its force manipulations. TODO: rethink the whole separate gathering launch.
 *
 * \param[in]  pme            The PME data structure.
 * \param[in]  wcycle         The wallclock counter.
 * \param[in,out] forces              The array of local atoms' resulting forces. Unless we do a reduction???
 * \param[in]  bClearForces   The boolean which tells whether the gathering kernel overwrites
 *                            the host array with the output PME forces (TRUE),
 *                            or copies its contents to the GPU and reduces the PME forces into that
 *                            (FALSE). The reduction is currently non-atomic.
 */
void pme_gpu_launch_gather(const gmx_pme_t      *pme,
                           gmx_wallcycle_t       wcycle,
                           rvec                 *forces,
                           bool                  bClearForces);

/*! \libinternal \brief
 * Gets the output forces and virial/energy if corresponding flags are (were?) passed in.
 *
 * \param[in]  pme            The PME data structure.
 * \param[in]  wcycle         The wallclock counter.
 * \param[out] vir_q          The output virial matrix.
 * \param[out] energy_q       The output energy.
 * \param[in]  flags          The combination of flags to affect the output.
 *                            Pass GMX_PME_CALC_ENER_VIR to get the virial and energy.
 *                            GMX_PME_CALC_F should be affecting the force output,
 *                            but likely will not as the force copy has already been scheduled before.
 *                            TODO: rethink the flag handling.
 */
void pme_gpu_get_results(const gmx_pme_t *pme,
                         gmx_wallcycle_t  wcycle,
                         matrix           vir_q,
                         real            *energy_q,
                         int              flags);

#endif // PMEGPU_H
