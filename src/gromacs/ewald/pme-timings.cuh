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
 *  \brief Defines the GPU timing event class in CUDA, and the PME GPU timing functions.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#ifndef PME_TIMINGS_CUH
#define PME_TIMINGS_CUH

#include "pme-internal.h"
#include "gromacs/timing/gpu_timing.h"

/*! \libinternal
 * \brief
 * This is a GPU timing class, based on CUDA events.
 *
 * Note that the data reported by CUDA events is not really reliable with multiple CUDA streams (e.g. PME and NB).
 * Therefore, it might be not that useful.
 *
 * \ingroup module_ewald
 */
class pme_gpu_timing
{
    bool         initialized;             /* Starts at false, set to true once */
    cudaEvent_t  event_start, event_stop; /* The internal timing events */
    unsigned int call_count;              /* Stars at 0, increased by stop_recording */
    real         total_milliseconds;      /* Starts at 0.0, increased by update */

    public:
        pme_gpu_timing();
        ~pme_gpu_timing();

        /*! \brief
         * To be called before the kernel/transfer launch.
         *
         * \param[in] s   The CUDA stream where the event being measured takes place.
         */
        void start_recording(cudaStream_t s);

        /*! \brief
         * To be called after the kernel/transfer launch.
         *
         * \param[in] s   The CUDA stream where the event being measured took place.
         */
        void stop_recording(cudaStream_t s);

        /*! \brief To be called after stop_recording and the CUDA stream of the event has been synchronised. */
        void update();

        // to be called once if needed
        void enable();
        void reset();
        real get_total_time_milliseconds();
        unsigned int get_call_count();
};

/*! \libinternal
 * \brief
 *
 * Allocates and initializes the PME GPU timings.
 *
 * \param[in] pme            The PME data structure.
 */
void pme_gpu_init_timings(const gmx_pme_t *pme);

/*! \libinternal
 * \brief
 *
 * Destroys the PME GPU timings.
 *
 * \param[in] pme            The PME data structure.
 */
void pme_gpu_destroy_timings(const gmx_pme_t *pme);

/*! \libinternal
 * \brief
 *
 * Starts timing the certain PME GPU stage during a single step (if timings are enabled).
 *
 * \param[in] pme            The PME data structure.
 * \param[in] PMEStageId     The PME GPU stage gtPME_ index from the enum in src/gromacs/timing/gpu_timing.h
 */
void pme_gpu_start_timing(const gmx_pme_t *pme, size_t PMEStageId);

/*! \libinternal
 * \brief
 *
 * Stops timing the certain PME GPU stage during a single step (if timings are enabled).
 *
 * \param[in] pme            The PME data structure.
 * \param[in] PMEStageId     The PME GPU stage gtPME_ index from the enum in src/gromacs/timing/gpu_timing.h
 */
void pme_gpu_stop_timing(const gmx_pme_t *pme, size_t PMEStageId);

/*! \libinternal
 * \brief
 *
 * Finalizes all the PEM GPU stage timings for the current step. Should be called at the end of every step.
 *
 * \param[in] pme            The PME data structure.
 */
void pme_gpu_update_timings(const gmx_pme_t *pme);

#endif
