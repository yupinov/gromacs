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
 *  \brief Implements PME GPU Fourier grid solving in CUDA.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include "pme-solve.h"

#include <memory>

#include "gromacs/ewald/pme-gpu.h"   //?
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/utility/gmxassert.h"

#include "pme.cuh"
#include "pme-3dfft.cuh"
#include "pme-internal.h"
#include "pme-timings.cuh"


template<const gmx_bool bEnerVir,
         // should the energy/virial be computed
         const gmx_bool YZXOrdering
         // false - GPU solve works in a XYZ ordering (after a single-rank cuFFT)
         // true - GPU solve works in a YZX ordering, like the CPU one (after FFTW)
         >
//__launch_bounds__(PME_SOLVE_THREADS_PER_BLOCK, PME_MIN_BLOCKS_PER_MP)
// TODO: figure out when this produces "invalid launch argument"
__global__ void pme_solve_kernel
    (const int localCountMajor, const int localCountMiddle, const int localCountMinor,
    const int localOffsetMinor, const int localOffsetMajor, const int localOffsetMiddle,
    const int localSizeMinor, /*const int localSizeMajor,*/ const int localSizeMiddle,
    const struct pme_gpu_cuda_kernel_params_t kernelParams
    )
{
    /* Global memory pointers */
    const float * __restrict__ splineValueMajorGlobal    = kernelParams.grid.d_splineModuli + kernelParams.grid.splineValuesOffset[YZXOrdering ? YY : XX];
    const float * __restrict__ splineValueMiddleGlobal   = kernelParams.grid.d_splineModuli + kernelParams.grid.splineValuesOffset[YZXOrdering ? ZZ : YY];
    const float * __restrict__ splineValueMinorGlobal    = kernelParams.grid.d_splineModuli + kernelParams.grid.splineValuesOffset[YZXOrdering ? XX : ZZ];
    float * __restrict__       virialAndEnergyGlobal     = kernelParams.constants.d_virialAndEnergy;


    // this is a PME solve kernel
    // each thread works on one cell of the Fourier space complex 3D grid (float2 * __restrict__ grid)
    // each block handles PME_SOLVE_THREADS_PER_BLOCK cells - depending on the grid contiguous dimension size,
    // that can range from a part of a single gridline to several complete gridlines
    // the minor dimension index is (YZXOrdering ? XX : ZZ)
    const int threadLocalId = (threadIdx.y * blockDim.x) + threadIdx.x;
    //const int blockSize = (blockDim.x * blockDim.y * blockDim.z); // == cellsPerBlock
    const int blockSize = PME_SOLVE_THREADS_PER_BLOCK;
    //const int threadId = blockId * blockSize + threadLocalId;

    float2 * __restrict__  globalGrid = (float2 *)kernelParams.grid.d_fourierGrid;

    const int              nMajor  = kernelParams.grid.localGridSize[YZXOrdering ? YY : XX];
    const int              nMiddle = kernelParams.grid.localGridSize[YZXOrdering ? ZZ : YY];
    const int              nMinor  = kernelParams.grid.localGridSize[YZXOrdering ? XX : ZZ]; //yupinov fix all pme->nkx and such

    int                    maxkMajor  = (nMajor + 1) / 2;                                    //X or Y
    int                    maxkMiddle = (nMiddle + 1) / 2;                                   //Y OR Z => only check for !YZX
    int                    maxkMinor  = (nMinor + 1) / 2;                                    //Z or X => only check for YZX

    float                  energy = 0.0f;
    float                  virxx  = 0.0f, virxy = 0.0f, virxz = 0.0f, viryy = 0.0f, viryz = 0.0f, virzz = 0.0f;

    const int              indexMinor  = blockIdx.x * blockDim.x + threadIdx.x;
    const int              indexMiddle = blockIdx.y * blockDim.y + threadIdx.y;
    const int              indexMajor  = blockIdx.z * blockDim.z + threadIdx.z;

    if ((indexMajor < localCountMajor) && (indexMiddle < localCountMiddle) && (indexMinor < localCountMinor))
    {
        /* The offset should be equal to the global thread index */
        float2     *globalGridPtr = globalGrid + (indexMajor * localSizeMiddle + indexMiddle) * localSizeMinor + indexMinor;

        const int   kMajor = indexMajor + localOffsetMajor;
        /* Checking either X in XYZ, or Y in YZX cases */
        const float mMajor = (kMajor < maxkMajor) ? kMajor : (kMajor - nMajor);

        const int   kMiddle = indexMiddle + localOffsetMiddle;
        float       mMiddle = kMiddle;
        /* Checking Y in XYZ case */
        if (!YZXOrdering)
        {
            mMiddle = (kMiddle < maxkMiddle) ? kMiddle : (kMiddle - nMiddle);
        }
        /* We should skip the k-space point (0,0,0) */
        const int       kMinor       = localOffsetMinor + indexMinor;
        const gmx_bool  notZeroPoint = (kMinor > 0 || kMajor > 0 || kMiddle > 0);
        float           mMinor       = kMinor, mhxk, mhyk, mhzk, m2k;

        /* Checking X in YZX case */
        if (YZXOrdering)
        {
            mMinor = (kMinor < maxkMinor) ? kMinor : (kMinor - nMinor);
        }

        float mX, mY, mZ;
        if (YZXOrdering)
        {
            mX = mMinor;
            mY = mMajor;
            mZ = mMiddle;
        }
        else
        {
            mX = mMajor;
            mY = mMiddle;
            mZ = mMinor;
        }

        /* 0.5 correction for corner points of a minor dimension */
        float corner_fac = 1.0f;
        if (YZXOrdering)
        {
            if (kMiddle == 0 || kMiddle == maxkMiddle)
            {
                corner_fac = 0.5f;
            }
        }
        else
        {
            if (kMinor == 0 || kMinor == maxkMinor)
            {
                corner_fac = 0.5f;
            }
        }

        if (notZeroPoint)
        {
            mhxk       = mX * kernelParams.step.recipBox[XX][XX];
            mhyk       = mX * kernelParams.step.recipBox[XX][YY] + mY * kernelParams.step.recipBox[YY][YY];
            mhzk       = mX * kernelParams.step.recipBox[XX][ZZ] + mY * kernelParams.step.recipBox[YY][ZZ] + mZ * kernelParams.step.recipBox[ZZ][ZZ];

            m2k        = mhxk * mhxk + mhyk * mhyk + mhzk * mhzk;
            float denom = m2k * float(M_PI) * kernelParams.step.boxVolume * splineValueMajorGlobal[kMajor] * splineValueMiddleGlobal[kMiddle] * splineValueMinorGlobal[kMinor];
            float tmp1  = -kernelParams.grid.ewaldFactor * m2k;

            denom = 1.0f / denom;
            tmp1  = expf(tmp1);
            float   etermk = kernelParams.constants.elFactor * tmp1 * denom;

            float2  gridValue    = *globalGridPtr;
            float2  oldGridValue = gridValue;
            gridValue.x   *= etermk;
            gridValue.y   *= etermk;
            *globalGridPtr = gridValue;

            if (bEnerVir)
            {
                float tmp1k = 2.0f * (gridValue.x * oldGridValue.x + gridValue.y * oldGridValue.y);

                float vfactor = (kernelParams.grid.ewaldFactor  + 1.0f / m2k) * 2.0f;
                float ets2    = corner_fac * tmp1k;
                energy = ets2;

                float ets2vf  = ets2 * vfactor;

                virxx   = ets2vf * mhxk * mhxk - ets2;
                virxy   = ets2vf * mhxk * mhyk;
                virxz   = ets2vf * mhxk * mhzk;
                viryy   = ets2vf * mhyk * mhyk - ets2;
                viryz   = ets2vf * mhyk * mhzk;
                virzz   = ets2vf * mhzk * mhzk - ets2;
            }
        }
    }

    if (bEnerVir)
    {
        /* The energy and virial reduction */

#if (GMX_PTX_ARCH >= 300)
        /* There really should be a shuffle reduction here!
         * (only for orders of power of 2)
         */
        /*
           if (!(blockSize & (blockSize - 1)))
           {

           }
           else
         */
#endif
        {
            __shared__ float virialAndEnergyShared[PME_GPU_VIRIAL_AND_ENERGY_COUNT * blockSize];
            // 3.5k smem per block - a serious limiter!

            /*  a 7-thread reduction in shared memory inspired by reduce_force_j_generic */
            if (threadLocalId < blockSize)
            {
                virialAndEnergyShared[threadLocalId + 0 * blockSize] = virxx;
                virialAndEnergyShared[threadLocalId + 1 * blockSize] = viryy;
                virialAndEnergyShared[threadLocalId + 2 * blockSize] = virzz;
                virialAndEnergyShared[threadLocalId + 3 * blockSize] = virxy;
                virialAndEnergyShared[threadLocalId + 4 * blockSize] = virxz;
                virialAndEnergyShared[threadLocalId + 5 * blockSize] = viryz;
                virialAndEnergyShared[threadLocalId + 6 * blockSize] = energy;
            }
            __syncthreads();

            /* Reducing every component to fit into warp_size */
            for (int s = blockSize >> 1; s >= warp_size; s >>= 1)
            {
#pragma unroll
                for (int i = 0; i < PME_GPU_VIRIAL_AND_ENERGY_COUNT; i++)
                {
                    if (threadLocalId < s) // split per threads?
                    {
                        virialAndEnergyShared[i * blockSize + threadLocalId] += virialAndEnergyShared[i * blockSize + threadLocalId + s];
                    }
                }
                __syncthreads();
            }

            const int threadsPerComponent    = warp_size / PME_GPU_VIRIAL_AND_ENERGY_COUNT; // this is also the stride, will be 32 / 7 = 4
            const int contributionsPerThread = warp_size / threadsPerComponent;             // will be 32 / 4 = 8
            if (threadLocalId < PME_GPU_VIRIAL_AND_ENERGY_COUNT * threadsPerComponent)
            {
                const int componentIndex        = threadLocalId / threadsPerComponent;
                const int threadComponentOffset = threadLocalId - componentIndex * threadsPerComponent;

                float     sum = 0.0f;
#pragma unroll
                for (int j = 0; j < contributionsPerThread; j++)
                {
                    sum += virialAndEnergyShared[componentIndex * blockSize + j * threadsPerComponent + threadComponentOffset];
                }
                atomicAdd(virialAndEnergyGlobal + componentIndex, sum);
            }

            /* A naive reduction */
            /*
               if (threadLocalId < blockSize)
               {
                virialAndEnergyShared[sizing * threadLocalId + 0] = virxx;
                virialAndEnergyShared[sizing * threadLocalId + 1] = viryy;
                virialAndEnergyShared[sizing * threadLocalId + 2] = virzz;
                virialAndEnergyShared[sizing * threadLocalId + 3] = virxy;
                virialAndEnergyShared[sizing * threadLocalId + 4] = virxz;
                virialAndEnergyShared[sizing * threadLocalId + 5] = viryz;
                virialAndEnergyShared[sizing * threadLocalId + 6] = energy;
               }
               __syncthreads();
               #pragma unroll
               for (unsigned int stride = 1; stride < blockSize; stride <<= 1)
               {
                if ((threadLocalId % (stride << 1) == 0))
                {
               #pragma unroll
                    for (int i = 0; i < sizing; i++)
                        virialAndEnergyShared[sizing * threadLocalId + i] += virialAndEnergyShared[sizing * (threadLocalId + stride) + i];
                }
                __syncthreads();
               }
               if (threadLocalId < sizing)
               {
                atomicAdd(virialAndEnergyGlobal + threadLocalId, virialAndEnergyShared[threadLocalId]);
               }
             */
        }
    }
}

void pme_gpu_solve(struct gmx_pme_t *pme, t_complex *grid, gmx_bool bEnerVir)
{
    /* do recip sum over local cells in grid */
    const pme_gpu_t *pmeGPU      = pme->gpu;
    const gmx_bool   YZXOrdering = !pme_gpu_performs_FFT(pmeGPU);
    /* true: y major, z middle, x minor or continuous - the CPU FFTW way */
    /* false: x major, y middle, z minor - the single rank GPU cuFFT way */
    //printf("ordering %d\n", YZXOrdering);

    cudaStream_t s = pmeGPU->archSpecific->pmeStream;
    const pme_gpu_cuda_kernel_params_t *kernelParamsPtr = pmeGPU->kernelParams.get();

    ivec         local_ndata, local_offset, local_size, complex_order;
    /* Dimensions should be identical for A/B grid, so we just use A here */

    if (pme_gpu_performs_FFT(pmeGPU))
    {
        pmeGPU->archSpecific->pfft_setup_gpu[PME_GRID_QA]->get_complex_limits(local_ndata, local_offset, local_size);
    }
    else
    {
        gmx_parallel_3dfft_complex_limits(pme->pfft_setup[PME_GRID_QA], complex_order, local_ndata, local_offset, local_size);
    }

    const int   minorDim  = !YZXOrdering ? ZZ : XX;
    const int   middleDim = !YZXOrdering ? YY : ZZ;
    const int   majorDim  = !YZXOrdering ? XX : YY;

    const int   gridSize = local_size[XX] * local_size[YY] * local_size[ZZ] * sizeof(float2);

    float2     *grid_d = (float2 *)kernelParamsPtr->grid.d_fourierGrid;
    if (!pme_gpu_performs_FFT(pmeGPU))
    {
        cu_copy_H2D_async(grid_d, grid, gridSize, s);
    }

    const int maxBlockSize      = bEnerVir ? PME_SOLVE_ENERVIR_THREADS_PER_BLOCK : PME_SOLVE_THREADS_PER_BLOCK;
    const int gridLineSize      = local_size[minorDim];
    const int gridLinesPerBlock = max(maxBlockSize / gridLineSize, 1);
    const int blocksPerGridLine = (gridLineSize + maxBlockSize - 1) / maxBlockSize; // rounded up
    // Z-dimension is too small in CUDA limitations (64 on CC30?), so instead of major-middle-minor sizing we do minor-middle-major
    dim3 threads((maxBlockSize + gridLinesPerBlock - 1) / gridLinesPerBlock, gridLinesPerBlock);
    const int blockSize = threads.x * threads.y * threads.z;
    GMX_RELEASE_ASSERT(blockSize >= maxBlockSize, "Wrong PME GPU solve launch parameters");
    // we want to be able to zero all the shared memory which we use in shared mem reduction

    dim3 blocks(blocksPerGridLine,
                (local_ndata[middleDim] + gridLinesPerBlock - 1) / gridLinesPerBlock, // rounded up middle dimension block number
                local_ndata[majorDim]);

    pme_gpu_start_timing(pmeGPU, gtPME_SOLVE);

    if (YZXOrdering)
    {
        if (bEnerVir)
        {
            pme_solve_kernel<TRUE, TRUE> <<< blocks, threads, 0, s>>>
            (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim],
             local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
             local_size[minorDim], /*local_size[majorDim],*/ local_size[middleDim],
             *kernelParamsPtr);
        }
        else
        {
            pme_solve_kernel<FALSE, TRUE> <<< blocks, threads, 0, s>>>
            (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim ],
             local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
             local_size[minorDim], /*local_size[majorDim],*/ local_size[middleDim],
             *kernelParamsPtr);
        }
    }
    else
    {
        if (bEnerVir)
        {
            pme_solve_kernel<TRUE, FALSE> <<< blocks, threads, 0, s>>>
            (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim],
             local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
             local_size[minorDim], /*local_size[majorDim],*/ local_size[middleDim],
             *kernelParamsPtr);
        }
        else
        {
            pme_solve_kernel<FALSE, FALSE> <<< blocks, threads, 0, s>>>
            (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim ],
             local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
             local_size[minorDim], /*local_size[majorDim],*/ local_size[middleDim],
             *kernelParamsPtr);
        }
    }
    CU_LAUNCH_ERR("pme_solve_kernel");

    pme_gpu_stop_timing(pmeGPU, gtPME_SOLVE);

    if (bEnerVir)
    {
        cu_copy_D2H_async(pmeGPU->staging.h_virialAndEnergy, kernelParamsPtr->constants.d_virialAndEnergy,
                          PME_GPU_VIRIAL_AND_ENERGY_COUNT * sizeof(float), s);
        cudaError_t stat = cudaEventRecord(pmeGPU->archSpecific->syncEnerVirD2H, s);
        CU_RET_ERR(stat, "PME solve energy/virial sync fail");
    }

    if (!pme_gpu_performs_FFT(pmeGPU))
    {
        cu_copy_D2H_async(grid, grid_d, gridSize, s);
        cudaError_t stat = cudaEventRecord(pmeGPU->archSpecific->syncSolveGridD2H, s);
        CU_RET_ERR(stat, "PME solve grid sync fail");
    }
}
