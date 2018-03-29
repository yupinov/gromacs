/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017, by the GROMACS development team, led by
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

/*! \internal \file
 *  \brief Implements OpenCL 3DFFT routines for PME GPU.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include <array>

#include "pme-3dfft-ocl.h"

#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

#include "gromacs/gpu_utils/oclutils.h"

//#include "pme.cuh"
#include "pme-types-ocl.h"

static void handleClfftError(clfftStatus status, const char *msg = nullptr)
{
  //suppsoedly it's just a superset of standard opencl errors
  throwUponFailure(status);
  // FIXME  if (status != CUFFT_SUCCESS)
  //  {
  //      gmx_fatal(FARGS, "%s (error code %d)\n", msg, status);
  //  }
}

GpuParallel3dFft::GpuParallel3dFft(const PmeGpu *pmeGpu)
{
    // Extracting all the data from PME GPU
    GMX_RELEASE_ASSERT(!pme_gpu_uses_dd(pmeGpu), "FFT decomposition not implemented");
    PmeGpuCudaKernelParams *kernelParamsPtr = pmeGpu->kernelParams.get();
    std::array<size_t, DIM> realGridSize, realGridSizePadded, complexGridSizePadded;
    for (int i = 0; i < DIM; i++)
    {
        realGridSize[i]          = kernelParamsPtr->grid.realGridSize[i];
        realGridSizePadded[i]    = kernelParamsPtr->grid.realGridSizePadded[i];
        complexGridSizePadded[i] = kernelParamsPtr->grid.complexGridSizePadded[i];
        GMX_ASSERT(kernelParamsPtr->grid.complexGridSizePadded[i] == kernelParamsPtr->grid.complexGridSize[i], "Complex padding not implemented");
    }
    cl_context context = pmeGpu->archSpecific->context;
    commandStreams_.push_back(pmeGpu->archSpecific->pmeStream);
    realGrid_ = kernelParamsPtr->grid.d_realGrid;
    complexGrid_ = kernelParamsPtr->grid.d_fourierGrid;
    const bool performOutOfPlaceFFT = pmeGpu->archSpecific->performOutOfPlaceFFT;

    // Setup
    clfftSetupData fftSetup;
    handleClfftError(clfftInitSetupData(&fftSetup), "clFFT data initialization failure");
    handleClfftError(clfftSetup(&fftSetup), "clFFT initialization failure");

    constexpr auto dims = CLFFT_3D;
    handleClfftError(clfftCreateDefaultPlan(&planR2C_, context, dims, realGridSize.data()), "clFFT planning failure");
    handleClfftError(clfftSetResultLocation(planR2C_, performOutOfPlaceFFT ? CLFFT_OUTOFPLACE : CLFFT_INPLACE), "clFFT planning failure");
    handleClfftError(clfftSetPlanPrecision(planR2C_, CLFFT_SINGLE), "clFFT planning failure"); //there is also CLFFT_SINGLE_FAST which is not implemented :/

    /*
    std::array<size_t, DIM> realGridStrides = {1, realGridSizePadded[YY], realGridSizePadded[XX] * realGridSizePadded[YY]};
    std::array<size_t, DIM> complexGridStrides = {1, complexGridSizePadded[YY], complexGridSizePadded[XX] * complexGridSizePadded[YY]};
    */

    /*
    std::array<size_t, DIM> realGridStrides = {realGridSizePadded[YY] * realGridSizePadded[ZZ], realGridSizePadded[ZZ], 1};
    std::array<size_t, DIM> complexGridStrides = {complexGridSizePadded[YY] * complexGridSizePadded[ZZ], complexGridSizePadded[ZZ], 1};
    */





    //printf ("grids %zu %zu %zu\n", realGridSizePadded[XX], realGridSizePadded[YY], realGridSizePadded[ZZ]);

    // THe only difference between 2 plans is direction
    handleClfftError(clfftCopyPlan(&planC2R_, context, planR2C_), "clFFT plan copying failure");
    handleClfftError(clfftSetLayout(planR2C_, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED), "clFFT R2C layout failure");
    handleClfftError(clfftSetLayout(planC2R_, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL), "clFFT C2R layout failure");

    printf ("real apdding ");
    for (int i =0; i < DIM; i++)
     printf ("%zu %zu ", realGridSize[i], realGridSizePadded[i] - realGridSize[i]);
    printf ("\n");

    printf ("complex apdding ");
    for (int i =0; i < DIM; i++)
     printf ("%zu %zu ", kernelParamsPtr->grid.complexGridSize[i], kernelParamsPtr->grid.complexGridSizePadded[i] - kernelParamsPtr->grid.complexGridSize[i]);
    printf ("\n");


    std::array<size_t, DIM> realGridStrides = {realGridSizePadded[YY] * realGridSizePadded[ZZ], realGridSizePadded[ZZ], 1};
    std::array<size_t, DIM> complexGridStrides = {complexGridSizePadded[YY] * complexGridSizePadded[ZZ], complexGridSizePadded[ZZ], 1};


    std::array<size_t, DIM> test;
    //handleClfftError(clfftGetPlanOutStride(planR2C_, dims, test.data()));
    //printf ("before  %zu %zu %zu\n", test[XX], test[YY], test[ZZ]);

    handleClfftError(clfftSetPlanInStride(planR2C_, dims, realGridStrides.data())); // TODO just use single plan instead?
    handleClfftError(clfftSetPlanOutStride(planR2C_, dims, complexGridStrides.data())); // TODO just use single plan instead?

    handleClfftError(clfftGetPlanInStride(planR2C_, dims, test.data()));
    printf ("after %zu %zu %zu real\n", test[XX], test[YY], test[ZZ]);
    handleClfftError(clfftGetPlanOutStride(planR2C_, dims, test.data()));
    printf ("after %zu %zu %zu complex\n", test[XX], test[YY], test[ZZ]);

    handleClfftError(clfftSetPlanInStride(planC2R_, dims, complexGridStrides.data())); // TODO just use single plan instead?
    handleClfftError(clfftSetPlanOutStride(planC2R_, dims, realGridStrides.data()));

    //TODO bake
#if 0






    const int complexGridSizePaddedTotal = complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ];
    const int realGridSizePaddedTotal    = realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ];

    /* Commented code for a simple 3D grid with no padding */
    /*
       result = cufftPlan3d(&planR2C_, realGridSize[XX], realGridSize[YY], realGridSize[ZZ], CUFFT_R2C);
       handleCufftError(result, "cufftPlan3d R2C plan failure");

       result = cufftPlan3d(&planC2R_, realGridSize[XX], realGridSize[YY], realGridSize[ZZ], CUFFT_C2R);
       handleCufftError(result, "cufftPlan3d C2R plan failure");
     */

    const int                 rank = 3, batch = 1;
    result = cufftPlanMany(&planR2C_, rank, realGridSize,
                           realGridSizePadded, 1, realGridSizePaddedTotal,
                           complexGridSizePadded, 1, complexGridSizePaddedTotal,
                           CUFFT_R2C,
                           batch);
    handleCufftError(result, "cufftPlanMany R2C plan failure");

    result = cufftPlanMany(&planC2R_, rank, realGridSize,
                           complexGridSizePadded, 1, complexGridSizePaddedTotal,
                           realGridSizePadded, 1, realGridSizePaddedTotal,
                           CUFFT_C2R,
                           batch);
    handleCufftError(result, "cufftPlanMany C2R plan failure");
#endif    
}

GpuParallel3dFft::~GpuParallel3dFft()
{
    //FIXME
    /*
    handleClfftError(clfftDestroyPlan(&planR2C_));
    handleClfftError(clfftDestroyPlan(&planC2R_));
    handleClfftError(clfftTeardown());
    */
}

void GpuParallel3dFft::perform3dFft(gmx_fft_direction dir)
{
    // Custom temp buffer could be interesting
    constexpr cl_mem tempBuffer = nullptr;
    constexpr std::array<cl_event, 0> waitEvents;
    constexpr cl_event *outEvents = nullptr;

    clfftPlanHandle plan;
    switch (dir)
    {
        case GMX_FFT_REAL_TO_COMPLEX:
        plan = planR2C_;
        break;

        break;
        case GMX_FFT_COMPLEX_TO_REAL:
        plan = planC2R_;
        break;

    default:
        GMX_ASSERT(false, "Not implemented");
        break;
    }

    handleClfftError(clfftEnqueueTransform(plan, CLFFT_FORWARD, //or backward?
                                           commandStreams_.size(), commandStreams_.data(),
                                           waitEvents.size(), waitEvents.data(), outEvents,
                                           &realGrid_, &complexGrid_, tempBuffer), "clFFT execution failure");
    //FIXME
}

//FIXME move to common
void pme_gpu_3dfft(const PmeGpu *pmeGpu, gmx_fft_direction dir, int grid_index)
{
    int timerId = (dir == GMX_FFT_REAL_TO_COMPLEX) ? gtPME_FFT_R2C : gtPME_FFT_C2R;
    pme_gpu_start_timing(pmeGpu, timerId);
    pmeGpu->archSpecific->fftSetup[grid_index]->perform3dFft(dir);
    pme_gpu_stop_timing(pmeGpu, timerId);
}
