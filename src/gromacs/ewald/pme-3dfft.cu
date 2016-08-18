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

/*! \internal \file
 *  \brief Implements CUDA FFT routines for PME GPU.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include <assert.h>
#include <cufft.h>
#include "pme-gpu.h"
#include "pme-timings.cuh"
#include "pme-cuda.cuh"

#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/gpu_utils/cudautils.cuh"

struct gmx_parallel_3dfft_gpu
{
    ivec          ndata_real;
    ivec          size_real;
    ivec          size_complex;

    cufftHandle   planR2C;
    cufftHandle   planC2R;
    cufftReal    *realGrid;
    cufftComplex *complexGrid;

    /* unused */
    ivec                      local_offset;
};

void gmx_parallel_3dfft_init_gpu(gmx_parallel_3dfft_gpu_t *pfft_setup, ivec ndata, gmx_pme_t *pme)
{
    cufftResult_t            result;
    gmx_parallel_3dfft_gpu_t setup;
    snew(setup, 1);

    setup->ndata_real[0] = ndata[XX];
    setup->ndata_real[1] = ndata[YY];
    setup->ndata_real[2] = ndata[ZZ];

    *pfft_setup = setup;

    if (pme->gpu->bGPUSingle)
    {
        ndata[XX] = pme->pmegrid_nx;
        ndata[YY] = pme->pmegrid_ny;
        ndata[ZZ] = pme->pmegrid_nz;
    }
    else
    {
        gmx_fatal(FARGS, "FFT size choice not implemented");
    }

    memcpy(setup->size_real, ndata, sizeof(setup->size_real));

    memcpy(setup->size_complex, setup->size_real, sizeof(setup->size_real));
    GMX_RELEASE_ASSERT(setup->size_complex[ZZ] % 2 == 0, "odd inplace cuFFT dimension size");
    setup->size_complex[ZZ] /= 2;
    // this is alright because Z includes overlap

    const int gridSizeComplex = setup->size_complex[XX] * setup->size_complex[YY] * setup->size_complex[ZZ];
    const int gridSizeReal    = setup->size_real[XX] * setup->size_real[YY] * setup->size_real[ZZ];

    memset(setup->local_offset, 0, sizeof(setup->local_offset)); //!

    setup->realGrid = (cufftReal *)pme->gpu->grid;
    assert(setup->realGrid);
    setup->complexGrid = (cufftComplex *)pme->gpu->fourierGrid;

    /*
       result = cufftPlan3d(&setup->planR2C, setup->ndata_real[XX], setup->ndata_real[YY], setup->ndata_real[ZZ], CUFFT_R2C);
       if (result != CUFFT_SUCCESS)
        gmx_fatal(FARGS, "cufftPlan3d R2C error %d\n", result);

       result = cufftPlan3d(&setup->planC2R, setup->ndata_real[XX], setup->ndata_real[YY], setup->ndata_real[ZZ], CUFFT_C2R);
       if (result != CUFFT_SUCCESS)
        gmx_fatal(FARGS, "cufftPlan3d C2R error %d\n", result);
     */


    const int rank = 3, batch = 1;
    result = cufftPlanMany(&setup->planR2C, rank, setup->ndata_real,
                           setup->size_real, 1, gridSizeReal,
                           setup->size_complex, 1, gridSizeComplex,
                           CUFFT_R2C,
                           batch);
    if (result != CUFFT_SUCCESS)
    {
        gmx_fatal(FARGS, "cufftPlanMany R2C error %d\n", result);
    }

    result = cufftPlanMany(&setup->planC2R, rank, setup->ndata_real,
                           setup->size_complex, 1, gridSizeComplex,
                           setup->size_real, 1, gridSizeReal,
                           CUFFT_C2R,
                           batch);
    if (result != CUFFT_SUCCESS)
    {
        gmx_fatal(FARGS, "cufftPlanMany C2R error %d\n", result);
    }

    cudaStream_t s = pme->gpu->pmeStream;
    assert(s);
    result = cufftSetStream(setup->planR2C, s);
    if (result != CUFFT_SUCCESS)
    {
        gmx_fatal(FARGS, "cufftSetStream R2C error %d\n", result);
    }

    result = cufftSetStream(setup->planC2R, s);
    if (result != CUFFT_SUCCESS)
    {
        gmx_fatal(FARGS, "cufftSetStream C2R error %d\n", result);
    }
}

void gmx_parallel_3dfft_real_limits_gpu(gmx_parallel_3dfft_gpu_t      setup,
                                        ivec                          local_ndata,
                                        ivec                          local_offset,
                                        ivec                          local_size)
{
    if (local_ndata)
    {
        memcpy(local_ndata, setup->ndata_real, sizeof(setup->ndata_real));
    }
    if (local_size)
    {
        memcpy(local_size, setup->size_real, sizeof(setup->size_real));
    }
    if (local_offset)
    {
        memcpy(local_offset, setup->local_offset, sizeof(setup->local_offset));
    }
}

void gmx_parallel_3dfft_complex_limits_gpu(gmx_parallel_3dfft_gpu_t      setup,
                                           ivec                          local_ndata,
                                           ivec                          local_offset,
                                           ivec                          local_size)
{
    if (local_ndata)
    {
        memcpy(local_ndata, setup->ndata_real, sizeof(setup->ndata_real));
        local_ndata[ZZ] = local_ndata[ZZ] / 2 + 1;
    }
    if (local_size)
    {
        memcpy(local_size, setup->size_complex, sizeof(setup->size_complex));
    }
    if (local_offset)
    {
        memcpy(local_offset, setup->local_offset, sizeof(setup->local_offset));
    }
}

void gmx_parallel_3dfft_execute_gpu(gmx_pme_t        *pme,
                                    gmx_fft_direction dir,
                                    const int         grid_index)
{
    /*
       const int gridSizeComplex = setup->size_complex[XX] * setup->size_complex[YY] * setup->size_complex[ZZ] * sizeof(cufftComplex);
       const int gridSizeReal = setup->size_real[XX] * setup->size_real[YY] * setup->size_real[ZZ] * sizeof(cufftReal);
     */
    gmx_parallel_3dfft_gpu_t setup = pme->gpu->pfft_setup_gpu[grid_index];

    if (dir == GMX_FFT_REAL_TO_COMPLEX)
    {
        //if (!pme->gpu->keepGPUDataBetweenSpreadAndR2C)
        //    cu_copy_H2D(setup->realGrid, setup->hostRealGrid, gridSizeReal);
        // CPU spread and GPU FFT? unlikely, only for debug

        pme_gpu_timing_start(pme, ewcsPME_FFT_R2C);

        cufftResult_t result = cufftExecR2C(setup->planR2C, setup->realGrid, setup->complexGrid);

        pme_gpu_timing_stop(pme, ewcsPME_FFT_R2C);

        if (result)
        {
            fprintf(stderr, "cufft R2C error %d\n", result);
        }
    }
    else
    {
        //if (!pme->gpu->keepGPUDataBetweenSolveAndC2R)
        //    cu_copy_H2D(setup->complexGrid, setup->hostComplexGrid, gridSizeComplex);
        // CPU solve and GPU FFT? unlikely, only for debug

        pme_gpu_timing_start(pme, ewcsPME_FFT_C2R);

        cufftResult_t result = cufftExecC2R(setup->planC2R, setup->complexGrid, setup->realGrid);

        pme_gpu_timing_stop(pme, ewcsPME_FFT_C2R);

        if (result)
        {
            fprintf(stderr, "cufft C2R error %d\n", result);
        }
    }
    /*
       if (dir == GMX_FFT_REAL_TO_COMPLEX)
       {
        // GPU FFT and CPU solve - unlikely, only for debug
        if (!pme->gpu->keepGPUDataBetweenR2CAndSolve)
            cu_copy_D2H(setup->hostComplexGrid, setup->complexGrid, gridSizeComplex);
       }
       else
       {
        //GPU FFT and CPU gather - unlikely, only for debug
        if (!pme->gpu->keepGPUDataBetweenC2RAndGather)
            cu_copy_D2H(setup->hostRealGrid, setup->realGrid, gridSizeReal);
       }
     */
}

void gmx_parallel_3dfft_destroy_gpu(const gmx_parallel_3dfft_gpu_t &pfft_setup)
{
    if (pfft_setup)
    {
        cufftResult_t result;

        result = cufftDestroy(pfft_setup->planR2C);
        if (result != CUFFT_SUCCESS)
        {
            gmx_fatal(FARGS, "cufftDestroy R2C error %d\n", result);
        }
        result = cufftDestroy(pfft_setup->planC2R);
        if (result != CUFFT_SUCCESS)
        {
            gmx_fatal(FARGS, "cufftDestroy C2R error %d\n", result);
        }

        sfree(pfft_setup);
    }
}
