/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2017, by the GROMACS development team, led by
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
#ifndef GMX_GPU_TUILS_CUDA_ERROR_HANDLERS_CUH
#define GMX_GPU_TUILS_CUDA_ERROR_HANDLERS_CUH

#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/stringutil.h"

/* TODO error checking needs to be rewritten. We have 2 types of error checks needed
   based on where they occur in the code:
   - non performance-critical: these errors are unsafe to be ignored and must be
     _always_ checked for, e.g. initializations
   - performance critical: handling errors might hurt performance so care need to be taken
     when/if we should check for them at all, e.g. in cu_upload_X. However, we should be
     able to turn the check for these errors on!

   Probably we'll need a few sets of the macros below...

   We also now have a separate set of always-throwing macros for testing purposes.
   The question is how applicable would those be performance-wise, can they replace the previous ones?
 */
#if 0
#define CHECK_CUDA_ERRORS

#ifdef CHECK_CUDA_ERRORS

#ifdef GMX_CUDA_THROW_ON_ERRORS
#pragma error this will no1 comppile 2
/*! Check for CUDA error on the return status of a CUDA RT API call. */
#define CU_RET_ERR(status, msg) \
    do { \
        if (status != cudaSuccess) \
        { \
            std::string error = gmx::formatString("%s: %s", msg, cudaGetErrorString(status)); \
            GMX_THROW(gmx::CudaError(error)); \
        } \
    } while (0)

/*! Check for any previously occurred uncaught CUDA error. */
#define CU_CHECK_PREV_ERR() \
    do { \
        cudaError_t _CU_CHECK_PREV_ERR_status = cudaGetLastError(); \
        if (_CU_CHECK_PREV_ERR_status != cudaSuccess) { \
            gmx_warning("Just caught a previously occurred CUDA error (%s), will try to continue.", cudaGetErrorString(_CU_CHECK_PREV_ERR_status)); \
        } \
    } while (0)

/*! Check for any previously occurred uncaught CUDA error
   -- aimed at use after kernel calls. */
#define CU_LAUNCH_ERR(msg) \
    do { \
        cudaError_t _CU_LAUNCH_ERR_status = cudaGetLastError(); \
        if (_CU_LAUNCH_ERR_status != cudaSuccess) { \
            gmx_fatal(FARGS, "Error while launching kernel %s: %s\n", msg, cudaGetErrorString(_CU_LAUNCH_ERR_status)); \
        } \
    } while (0)

/*! Synchronize with GPU and check for any previously occurred uncaught CUDA error
   -- aimed at use after kernel calls. */
#define CU_LAUNCH_ERR_SYNC(msg) \
    do { \
        cudaError_t _CU_SYNC_LAUNCH_ERR_status = cudaThreadSynchronize(); \
        if (_CU_SYNC_LAUNCH_ERR_status != cudaSuccess) { \
            gmx_fatal(FARGS, "Error while launching kernel %s: %s\n", msg, cudaGetErrorString(_CU_SYNC_LAUNCH_ERR_status)); \
        } \
    } while (0)

#else //!defined(GMX_CUDA_THROW_ON_ERRORS)

/*! Check for CUDA error on the return status of a CUDA RT API call. */
#define CU_RET_ERR(status, msg) \
    do { \
        if (status != cudaSuccess) \
        { \
            gmx_fatal(FARGS, "%s: %s\n", msg, cudaGetErrorString(status)); \
        } \
    } while (0)

/*! Check for any previously occurred uncaught CUDA error. */
#define CU_CHECK_PREV_ERR() \
    do { \
        cudaError_t _CU_CHECK_PREV_ERR_status = cudaGetLastError(); \
        if (_CU_CHECK_PREV_ERR_status != cudaSuccess) { \
            gmx_warning("Just caught a previously occurred CUDA error (%s), will try to continue.", cudaGetErrorString(_CU_CHECK_PREV_ERR_status)); \
        } \
    } while (0)

/*! Check for any previously occurred uncaught CUDA error
   -- aimed at use after kernel calls. */
#define CU_LAUNCH_ERR(msg) \
    do { \
        cudaError_t _CU_LAUNCH_ERR_status = cudaGetLastError(); \
        if (_CU_LAUNCH_ERR_status != cudaSuccess) { \
            gmx_fatal(FARGS, "Error while launching kernel %s: %s\n", msg, cudaGetErrorString(_CU_LAUNCH_ERR_status)); \
        } \
    } while (0)

/*! Synchronize with GPU and check for any previously occurred uncaught CUDA error
   -- aimed at use after kernel calls. */
#define CU_LAUNCH_ERR_SYNC(msg) \
    do { \
        cudaError_t _CU_SYNC_LAUNCH_ERR_status = cudaThreadSynchronize(); \
        if (_CU_SYNC_LAUNCH_ERR_status != cudaSuccess) { \
            gmx_fatal(FARGS, "Error while launching kernel %s: %s\n", msg, cudaGetErrorString(_CU_SYNC_LAUNCH_ERR_status)); \
        } \
    } while (0)
#endif
#else /* CHECK_CUDA_ERRORS */

#define CU_RET_ERR(status, msg) do { } while (0)
#define CU_CHECK_PREV_ERR()     do { } while (0)
#define CU_LAUNCH_ERR(msg)      do { } while (0)
#define CU_LAUNCH_ERR_SYNC(msg) do { } while (0)
#define HANDLE_NVML_RET_ERR(status, msg) do { } while (0)

#endif /* CHECK_CUDA_ERRORS */
#endif

namespace gmx
{

namespace CudaErrorHandling
{

// TODO: put cufft/NVML here as well

//! How often(???) do we check the CUDA return codes
enum class CheckMode
{
    All,       // test mode
    Sometimes, // do  not harm performance
    Never      // why not?
};

enum class ErrorType //do we need it??
{
    General,
    Cufft,
    Nvml,
};

enum class HandlingMode
{
    OldStyle,   // gmx_fatal/warning flavors - abd for testing
    Exceptions, // flexible, but what about performance
};

template <CheckMode, HandlingMode>
class Handler
{
    public:
        void checkForError(cudaError_t status, const char *message);
};


//old style, check
/*
   void checkForError(cudaError_t status, const char *message)
   {
    if (status != cudaSuccess)
    {
        gmx_fatal(FARGS, "%s: %s\n", message, cudaGetErrorString(status));
    }
   }
 */

extern Handler *c_CudaHandler;

#define CHECK_CUDA_ERRORS 1
#if CHECK_CUDA_ERRORS
c_CudaHandler = new Handler<CheckMode::All, HandlingMode::OldStyle>;  //TODO delete?
#endif

#define CU_RET_ERR(status, msg) do { \
} while (0)
#define CU_CHECK_PREV_ERR()     do { } while (0)
#define CU_LAUNCH_ERR(msg)      do { } while (0)
#define CU_LAUNCH_ERR_SYNC(msg) do { } while (0)
#define HANDLE_NVML_RET_ERR(status, msg) do { } while (0)


}
}
#endif // GMX_GPU_TUILS_CUDA_ERROR_HANDLERS_CUH
