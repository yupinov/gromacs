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
#ifndef GMX_GPU_UTILS_DEVICEBUFFER_OCL_H
#define GMX_GPU_UTILS_DEVICEBUFFER_OCL_H

/*! \libinternal \file
 *  \brief Implements the DeviceBuffer type and routines for OpenCL.
 *  TODO: the intent is for DeviceBuffer to become a class.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *
 *  \inlibraryapi
 */

#include "gromacs/gpu_utils/gpu_utils.h" //only for GpuApiCallBehavior
#include "gromacs/gpu_utils/gputraits_ocl.h"
#include "gromacs/utility/gmxassert.h"

/*! \libinternal \brief
 * A minimal cl_mem wrapper that remembers its allocation type.
 * The only point is making template type deduction possible.
 */
template<typename ValueType>
class TypedClMemory
{
    private:
        //! \brief Underlying data - not nulled right here only because we still have some snew()'s around
        cl_mem data_;
    public:
        //! \brief An assignment operator - the purpose is to make allocation/zeroing work
        void operator=(void *ptr){data_ = (cl_mem)ptr; }
        //! \brief Returns underlying cl_mem transparently
        operator cl_mem() {return data_; }
};

//! \libinternal \brief A device-side buffer of ValueTypes
template<typename ValueType>
using DeviceBuffer = TypedClMemory<ValueType>;

/*! \libinternal \brief
 * Allocates a device-side buffer.
 * It is currently a caller's responsibility to call it only on not-yet allocated buffers.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] buffer               Pointer to the device-side buffer.
 * \param[in]     numValues            Number of values to accomodate.
 * \param[in]     context              The buffer's context-to-be.
 */
template <typename ValueType>
void allocateDeviceBuffer(DeviceBuffer<ValueType> *buffer,
                          size_t                   numValues,
                          Context                  context)
{
    GMX_ASSERT(buffer, "needs a buffer pointer");
    void  *hostPtr = nullptr;
    cl_int clError;
    *buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, numValues * sizeof(ValueType), hostPtr, &clError);
    GMX_RELEASE_ASSERT(clError == CL_SUCCESS, "clCreateBuffer failure");
}

/*! \brief
 * Frees a device-side buffer.
 * This does not reset separately stored size/capacity integers,
 * as this is planned to be a destructor of DeviceBuffer as a proper class,
 * and no calls on \p buffer should be made afterwards.
 *
 * \param[in] buffer  Pointer to the buffer to free.
 */
template <typename DeviceBuffer>
void freeDeviceBuffer(DeviceBuffer *buffer)
{
    GMX_ASSERT(buffer, "needs a buffer pointer");
    if (*buffer)
    {
        GMX_RELEASE_ASSERT(clReleaseMemObject(*buffer) == CL_SUCCESS, "clReleaseMemObject failed");
    }
}

/*! \brief
 * Performs the host-to-device data copy, synchronous or asynchronously on request.
 *
 * TODO: This is meant to gradually replace cu/ocl_copy_h2d.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] buffer               Pointer to the device-side buffer
 * \param[in]     hostBuffer           Pointer to the raw host-side memory, also typed \p ValueType
 * \param[in]     startingValueIndex   Offset (in values) at the device-side buffer to copy into.
 * \param[in]     numValues            Number of values to copy.
 * \param[in]     stream               GPU stream to perform asynchronous copy in.
 * \param[in]     transferKind         Copy type: synchronous or asynchronous.
 * \param[out]    timingEvent          A pointer to the H2D copy timing event to be filled in.
 *                                     If the pointer is not null, the event can further be used
 *                                     to queue a wait for this operation or to query profiling information.
 */
template <typename ValueType>
void copyToDeviceBuffer(DeviceBuffer<ValueType> *buffer,
                        const ValueType         *hostBuffer,
                        size_t                   startingValueIndex,
                        size_t                   numValues,
                        CommandStream            stream,
                        GpuApiCallBehavior       transferKind,
                        CommandEvent            *timingEvent)
{
    if (numValues == 0)
    {
        return; // such calls are actually made with empty domains
    }
    GMX_ASSERT(buffer, "needs a buffer pointer");
    GMX_ASSERT(hostBuffer, "needs a host buffer pointer");
    cl_int       clError;
    const size_t offset = startingValueIndex * sizeof(ValueType);
    const size_t bytes  = numValues * sizeof(ValueType);
    switch (transferKind)
    {
        case GpuApiCallBehavior::Async:
            clError = clEnqueueWriteBuffer(stream, *buffer, CL_FALSE, offset, bytes, hostBuffer, 0, nullptr, timingEvent);
            //GMX_RELEASE_ASSERT(clError == CL_SUCCESS, "Asynchronous H2D copy failed");
            break;

        case GpuApiCallBehavior::Sync:
            clError = clEnqueueWriteBuffer(stream, *buffer, CL_TRUE, offset, bytes, hostBuffer, 0, nullptr, timingEvent);
            //GMX_RELEASE_ASSERT(clError == CL_SUCCESS, "Synchronous H2D copy failed");
            break;

        default:
            throw;
    }
    throwUponFailure(clError);
}

/*! \brief
 * Performs the device-to-host data copy, synchronous or asynchronously on request.
 *
 * TODO: This is meant to gradually replace cu/ocl_copy_d2h.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] hostBuffer           Pointer to the raw host-side memory, also typed \p ValueType
 * \param[in]     buffer               Pointer to the device-side buffer
 * \param[in]     startingValueIndex   Offset (in values) at the device-side buffer to copy from.
 * \param[in]     numValues            Number of values to copy.
 * \param[in]     stream               GPU stream to perform asynchronous copy in.
 * \param[in]     transferKind         Copy type: synchronous or asynchronous.
 * \param[out]    timingEvent          A pointer to the H2D copy timing event to be filled in.
 *                                     If the pointer is not null, the event can further be used
 *                                     to queue a wait for this operation or to query profiling information.
 */
template <typename ValueType>
void copyFromDeviceBuffer(ValueType                     *hostBuffer,
                          DeviceBuffer<ValueType>       *buffer,
                          size_t                         startingValueIndex,
                          size_t                         numValues,
                          CommandStream                  stream,
                          GpuApiCallBehavior             transferKind,
                          CommandEvent                  *timingEvent)
{
    GMX_ASSERT(buffer, "needs a buffer pointer");
    GMX_ASSERT(hostBuffer, "needs a host buffer pointer");
    cl_int       clError;
    const size_t offset = startingValueIndex * sizeof(ValueType);
    const size_t bytes  = numValues * sizeof(ValueType);
    switch (transferKind)
    {
        case GpuApiCallBehavior::Async:
            clError = clEnqueueReadBuffer(stream, *buffer, CL_FALSE, offset, bytes, hostBuffer, 0, nullptr, timingEvent);
            GMX_RELEASE_ASSERT(clError == CL_SUCCESS, "Asynchronous D2H copy failed");
            break;

        case GpuApiCallBehavior::Sync:
            clError = clEnqueueReadBuffer(stream, *buffer, CL_TRUE, offset, bytes, hostBuffer, 0, nullptr, timingEvent);
            GMX_RELEASE_ASSERT(clError == CL_SUCCESS, "Synchronous D2H copy failed");
            break;

        default:
            throw;
    }
}

/*! \brief
 * Clears the device buffer asynchronously. Assumes OpenCL 1.2.
 *
 * \tparam        ValueType            Raw value type of the \p buffer.
 * \param[in,out] buffer               Pointer to the device-side buffer
 * \param[in]     startingValueIndex   Offset (in values) at the device-side buffer to start clearing at.
 * \param[in]     numValues            Number of values to clear.
 * \param[in]     stream               GPU stream.
 */
template <typename ValueType>
void clearDeviceBufferAsync(DeviceBuffer<ValueType> *buffer,
                            size_t                   startingValueIndex,
                            size_t                   numValues,
                            CommandStream            stream)
{
   GMX_ASSERT(buffer, "needs a buffer pointer");
   const size_t offset = startingValueIndex * sizeof(ValueType);
   const size_t bytes  = numValues * sizeof(ValueType);
   const ValueType pattern = 0;
   const cl_uint numWaitEvents = 0;
   const cl_event *waitEvents = nullptr;
   cl_event commandEvent;
   cl_int clError = clEnqueueFillBuffer(stream, *buffer, &pattern, sizeof(pattern),
                                        offset, bytes,
                                        numWaitEvents, waitEvents, &commandEvent);
   GMX_RELEASE_ASSERT(clError == CL_SUCCESS, "Couldn't clear the device buffer");
}

//TEtxure table

/*! \brief Initialize parameter lookup table.
 * //FIXME make it use DeviceBuffer wrapper!
 *
 * Initializes device memory, copies data from host and binds
 * a texture to allocated device memory to be used for parameter lookup.
 *
 * \tparam[in] ValueType Raw data type
 * \param[out] d_ptr     device pointer to the memory to be allocated
 * \param[out] texObj    texture object to be initialized
 * \param[in]  h_ptr     pointer to the host memory to be uploaded to the device
 * \param[in]  numElem   number of elements in the h_ptr
 * \param[in]  devInfo   pointer to the info struct of the device in use
 * \param[in]  stream    GPU stream for copying //FIXME
 */
template <typename ValueType>
void initParamLookupTable(DeviceBuffer<ValueType>   *buffer,
                          int                     &dummyCudaTexObj,
                          const ValueType            *hostBuffer,
                          size_t                      numValues,
                          const gmx_device_info_t    *devInfo,
                          Context                     context,
			  CommandStream               stream)
{
    GMX_ASSERT(buffer, "needs a buffer pointer");
    allocateDeviceBuffer(buffer, numValues, context);
    GMX_UNUSED_VALUE(dummyCudaTexObj);
    GMX_UNUSED_VALUE(devInfo);
    copyToDeviceBuffer(buffer, hostBuffer, 0, numValues, stream, GpuApiCallBehavior::Sync, nullptr);
    //FIXME  c_disableCudaTextures
}

/*! \brief Destroy parameter lookup table.
 *
 * Unbinds texture object, deallocates device memory.
 *
 * \tparam[in] ValueType Raw data type
 * \param[in]  d_ptr     Device pointer to the memory to be deallocated
 * \param[in]  texObj    Texture object to be deinitialized
 * \param[in]  devInfo   Pointer to the info struct of the device in use
 */
template <typename ValueType>
void destroyParamLookupTable(DeviceBuffer<ValueType>   *buffer,
                             int                     &dummyCudaTexObj,
                             const gmx_device_info_t *devInfo)
{
    GMX_ASSERT(buffer, "needs a buffer pointer");
    GMX_UNUSED_VALUE(dummyCudaTexObj);
    GMX_UNUSED_VALUE(devInfo);
    freeDeviceBuffer(buffer);
}

// included after DeviceBuffer and its implementation functions are defined
#include "gromacs/gpu_utils/devicebuffer.h"

#endif
