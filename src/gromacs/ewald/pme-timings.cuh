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
#ifndef PME_TIMINGS_CUH
#define PME_TIMINGS_CUH

#include "pme-internal.h"
#include "gromacs/timing/gpu_timing.h"

class pme_gpu_timing
{
    bool         initialized;
    cudaEvent_t  event_start, event_stop;
    unsigned int call_count;
    real         total_milliseconds;

    public:
        pme_gpu_timing();
        ~pme_gpu_timing();

        // to be called every MD step if needed
        void start_recording(cudaStream_t s);
        void stop_recording(cudaStream_t s);
        void update();

        // to be called once if needed
        void enable();
        void reset();
        real get_total_time_milliseconds();
        unsigned int get_call_count();
};

void pme_gpu_init_timings(gmx_pme_t *pme);
void pme_gpu_timing_start(gmx_pme_t *pme, int PMEStageId);
void pme_gpu_timing_stop(gmx_pme_t *pme, int PMEStageId);
void pme_gpu_update_timings(gmx_pme_t *pme);
void pme_gpu_destroy_timings(gmx_pme_t *pme);

#endif
