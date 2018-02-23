/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018, by the GROMACS development team, led by
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
#ifndef GMX_GPU_UTILS_GPUTRAITS_OCL_H
#define GMX_GPU_UTILS_GPUTRAITS_OCL_H

/*! \libinternal \file
 *  \brief Declares the OpenCL type traits.
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *
 * \inlibraryapi
 */

#include "gromacs/gpu_utils/gmxopencl.h"

//! \brief GPU command stream
using CommandStream = cl_command_queue;
//! \brief Single GPU call timing event
using CommandEvent  = cl_event;
//! \brief Context used explicitly in OpenCL
using Context       = cl_context;

//Sync object

#include "gromacs/utility/gmxassert.h"
#include "gromacs/gpu_utils/oclutils.h" //for ocl_get_error_string


class SyncEvent
{
    public:
        SyncEvent()
        {
            //??????
        }

        ~SyncEvent()
        {
            cl_int clError = clReleaseEvent(event_);
            GMX_RELEASE_ASSERT(CL_SUCCESS == clError, ocl_get_error_string(clError).c_str());
        }

        //FIXME disable copy

        inline void markSyncEvent(CommandStream stream)
        {
            cl_int clError;
#ifdef CL_VERSION_1_2
            clError = clEnqueueMarkerWithWaitList(stream, 0, nullptr, &event_);
#else
            clError = clEnqueueMarker(stream, &event_);
#endif
            GMX_ASSERT(CL_SUCCESS == clError, ocl_get_error_string(clError).c_str());
        }

        /*! \brief Enqueues a wait for event completion.
         *
         * Then it releases the event and sets it to 0.
         * Don't use this function when more than one wait will be issued for the event.
         * Equivalent to Cuda Stream Sync.
        */
        // copied from sync_ocl_event
        inline void waitForSyncEvent(CommandStream stream)
        {
            cl_int clError;

            /* Enqueue wait */
        #ifdef CL_VERSION_1_2
            clError = clEnqueueBarrierWithWaitList(stream, 1, &event_, nullptr);
        #else
            clEerror = clEnqueueWaitForEvents(stream, 1, &event_);
        #endif
            GMX_ASSERT(CL_SUCCESS == clError, ocl_get_error_string(clError).c_str());

            /* Release event and reset it to 0. It is ok to release it as enqueuewaitforevents performs implicit retain for events. */
            clError = clReleaseEvent(event_);
            GMX_ASSERT(CL_SUCCESS == clError, ocl_get_error_string(clError).c_str());
            event_ = nullptr; //FIXME is thsi correct&
        }

    private:
        cl_event event_;
};

#endif
