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
 *  \brief Implements PME GPU Fourier grid solving in CUDA.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include "config.h"

#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"

#include "pme.cuh"
#include "pme-timings.cuh"

//! Tested on 560Ti (CC2.1), 660Ti (CC3.0) and 750 (CC5.0) GPUs (among 64, 128, 256, 512, 1024)
constexpr int PME_SOLVE_THREADS_PER_BLOCK = (8 * warp_size);

// CUDA 6.5 can not compile enum class as a template kernel parameter,
// so we replace it with a duplicate simple enum
#if GMX_CUDA_VERSION >= 7000
using GridOrderingInternal = GridOrdering;
#else
enum GridOrderingInternal
{
    YZX,
    XYZ
};
#endif

/*! \brief
 * PME complex grid solver kernel function.
 *
 * \tparam[in] computeEnergyAndVirial   Tells if the reciprocal energy and virial should be computed.
 * \tparam[in] gridOrdering             Specifies the dimension ordering of the complex grid.
 * \param[in]  kernelParams             Input PME CUDA data in constant memory.
 */
template<
    bool computeEnergyAndVirial,
    GridOrderingInternal gridOrdering
    >
__launch_bounds__(PME_SOLVE_THREADS_PER_BLOCK, PME_MIN_BLOCKS_PER_MP) //FIXME use the max number per arch
__global__ void pme_solve_kernel(const struct pme_gpu_cuda_kernel_params_t kernelParams)
{
    /* This kernel supports 2 different grid dimension orderings: YZX and XYZ */
    int majorDim, middleDim, minorDim;
    switch (gridOrdering)
    {
        case GridOrderingInternal::YZX:
            majorDim  = YY;
            middleDim = ZZ;
            minorDim  = XX;
            break;

        case GridOrderingInternal::XYZ:
            majorDim  = XX;
            middleDim = YY;
            minorDim  = ZZ;
            break;

        default:
            assert(false);
    }

    /* Global memory pointers */
    const float * __restrict__ gm_splineValueMajor    = kernelParams.grid.d_splineModuli + kernelParams.grid.splineValuesOffset[majorDim];
    const float * __restrict__ gm_splineValueMiddle   = kernelParams.grid.d_splineModuli + kernelParams.grid.splineValuesOffset[middleDim];
    const float * __restrict__ gm_splineValueMinor    = kernelParams.grid.d_splineModuli + kernelParams.grid.splineValuesOffset[minorDim];
    float * __restrict__       gm_virialAndEnergy     = kernelParams.constants.d_virialAndEnergy;
    float2 * __restrict__      gm_grid                = (float2 *)kernelParams.grid.d_fourierGrid;

    /* Various grid sizes and indices */
    const int localOffsetMinor = 0, localOffsetMajor = 0, localOffsetMiddle = 0; //unused
    const int localSizeMinor   = kernelParams.grid.complexGridSizePadded[minorDim];
    const int localSizeMiddle  = kernelParams.grid.complexGridSizePadded[middleDim];
    const int localCountMajor  = kernelParams.grid.complexGridSize[majorDim];
    const int localCountMiddle = kernelParams.grid.complexGridSize[middleDim];
    const int localCountMinor  = kernelParams.grid.complexGridSize[minorDim];
    const int nMajor           = kernelParams.grid.realGridSize[majorDim];
    const int nMiddle          = kernelParams.grid.realGridSize[middleDim];
    const int nMinor           = kernelParams.grid.realGridSize[minorDim];
    const int maxkMajor        = (nMajor + 1) / 2;  // X or Y
    const int maxkMiddle       = (nMiddle + 1) / 2; // Y OR Z => only check for !YZX
    const int maxkMinor        = (nMinor + 1) / 2;  // Z or X => only check for YZX

    /* Each thread works on one cell of the Fourier space complex 3D grid (gm_grid).
     * Each block handles PME_SOLVE_THREADS_PER_BLOCK cells -
     * depending on the grid contiguous dimension size,
     * that can range from a part of a single gridline to several complete gridlines.
     */
    const int threadLocalId = (threadIdx.y * blockDim.x) + threadIdx.x;
    const int indexMinor    = blockIdx.x * blockDim.x + threadIdx.x;
    const int indexMiddle   = blockIdx.y * blockDim.y + threadIdx.y;
    const int indexMajor    = blockIdx.z * blockDim.z + threadIdx.z;

    /* Optional outputs */
    float energy = 0.0f;
    float virxx  = 0.0f;
    float virxy  = 0.0f;
    float virxz  = 0.0f;
    float viryy  = 0.0f;
    float viryz  = 0.0f;
    float virzz  = 0.0f;

    if ((indexMajor < localCountMajor) & (indexMiddle < localCountMiddle) & (indexMinor < localCountMinor))
    {
        /* The offset should be equal to the global thread index for coalesced access */
        const int            gridIndex     = (indexMajor * localSizeMiddle + indexMiddle) * localSizeMinor + indexMinor;
        float2 __restrict__ *gm_gridCell   = gm_grid + gridIndex;
        // TODO possibly reuse inlined indexing function from tests

        const int   kMajor = indexMajor + localOffsetMajor;
        /* Checking either X in XYZ, or Y in YZX cases */
        const float mMajor = (kMajor < maxkMajor) ? kMajor : (kMajor - nMajor);

        const int   kMiddle = indexMiddle + localOffsetMiddle;
        float       mMiddle = kMiddle;
        /* Checking Y in XYZ case */
        if (gridOrdering == GridOrderingInternal::XYZ)
        {
            mMiddle = (kMiddle < maxkMiddle) ? kMiddle : (kMiddle - nMiddle);
        }
        const int       kMinor        = localOffsetMinor + indexMinor;
        float           mMinor        = kMinor;
        /* Checking X in YZX case */
        if (gridOrdering == GridOrderingInternal::YZX)
        {
            mMinor = (kMinor < maxkMinor) ? kMinor : (kMinor - nMinor);
        }
        /* We should skip the k-space point (0,0,0) */
        const bool notZeroPoint  = (kMinor > 0) | (kMajor > 0) | (kMiddle > 0);

        float      mX, mY, mZ;
        switch (gridOrdering)
        {
            case GridOrderingInternal::YZX:
                mX = mMinor;
                mY = mMajor;
                mZ = mMiddle;
                break;

            case GridOrderingInternal::XYZ:
                mX = mMajor;
                mY = mMiddle;
                mZ = mMinor;
                break;

            default:
                assert(false);
        }

        /* 0.5 correction factor for the first and last components of a minor dimension */
        float corner_fac = 1.0f;
        switch (gridOrdering)
        {
            case GridOrderingInternal::YZX:
                if ((kMiddle == 0) | (kMiddle == maxkMiddle))
                {
                    corner_fac = 0.5f; //FIXME this actually depends on the FFT minor dim - should probably always take minor? Test with FFTW
                }
                break;

            case GridOrderingInternal::XYZ:
                if ((kMinor == 0) | (kMinor == maxkMinor))
                {
                    corner_fac = 0.5f;
                }
                break;

            default:
                assert(false);
        }

        if (notZeroPoint)
        {
            const float mhxk = mX * kernelParams.step.recipBox[XX][XX];
            const float mhyk = mX * kernelParams.step.recipBox[XX][YY] + mY * kernelParams.step.recipBox[YY][YY];
            const float mhzk = mX * kernelParams.step.recipBox[XX][ZZ] + mY * kernelParams.step.recipBox[YY][ZZ] + mZ * kernelParams.step.recipBox[ZZ][ZZ];

            const float m2k        = mhxk * mhxk + mhyk * mhyk + mhzk * mhzk;
            assert(m2k != 0.0f);
            float       denom = m2k * float(M_PI) * kernelParams.step.boxVolume * gm_splineValueMajor[kMajor] * gm_splineValueMiddle[kMiddle] * gm_splineValueMinor[kMinor];
            assert(!isnan(denom));
            assert(denom != 0.0f);
            const float   tmp1   = expf(-kernelParams.grid.ewaldFactor * m2k);
            const float   etermk = kernelParams.constants.elFactor * tmp1 / denom;

            float2        gridValue    = *gm_gridCell;
            const float2  oldGridValue = gridValue;
            gridValue.x   *= etermk;
            gridValue.y   *= etermk;
            *gm_gridCell   = gridValue;

            if (computeEnergyAndVirial)
            {
                const float tmp1k = 2.0f * (gridValue.x * oldGridValue.x + gridValue.y * oldGridValue.y);

                float       vfactor = (kernelParams.grid.ewaldFactor + 1.0f / m2k) * 2.0f;
                float       ets2    = corner_fac * tmp1k;
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

    /* Optional energy/virial reduction */
    if (computeEnergyAndVirial)
    {
#if (GMX_PTX_ARCH >= 300)
        /* A tricky shuffle reduction inspired by reduce_force_j_warp_shfl.
         * The idea is to reduce 7 energy/virial components into a single variable (aligned by 8).
         * We will reduce everything into virxx.
         */

        /* We can only reduce warp-wise */
        const int width = warp_size;

        /* Making pair sums */
        virxx  += __shfl_down(virxx, 1, width);
        viryy  += __shfl_up  (viryy, 1, width);
        virzz  += __shfl_down(virzz, 1, width);
        virxy  += __shfl_up  (virxy, 1, width);
        virxz  += __shfl_down(virxz, 1, width);
        viryz  += __shfl_up  (viryz, 1, width);
        energy += __shfl_down(energy, 1, width);
        if (threadLocalId & 1)
        {
            virxx = viryy; // virxx now holds virxx and viryy pair sums
            virzz = virxy; // virzz now holds virzz and virxy pair sums
            virxz = viryz; // virxz now holds virxz and viryz pair sums
        }

        /* Making quad sums */
        virxx  += __shfl_down(virxx, 2, width);
        virzz  += __shfl_up  (virzz, 2, width);
        virxz  += __shfl_down(virxz, 2, width);
        energy += __shfl_up(energy, 2, width);
        if (threadLocalId & 2)
        {
            virxx = virzz;  // virxx now holds quad sums of virxx, virxy, virzz and virxy
            virxz = energy; // virxz now holds quad sums of virxz, viryz, energy and unused paddings
        }

        /* Making octet sums */
        virxx += __shfl_down(virxx, 4, width);
        virxz += __shfl_up(virxz, 4, width);
        if (threadLocalId & 4)
        {
            virxx = virxz; // virxx now holds all 7 components' octets sums + unused paddings
        }

        /* We only need to reduce virxx now */
#pragma unroll
        for (int delta = 8; delta < width; delta <<= 1)
        {
            virxx += __shfl_down(virxx, delta, width);
        }
        /* Now first 7 threads of each warp have the full output contributions in virxx */

        const int  componentIndex      = threadLocalId & 0x1f;
        const bool validComponentIndex = (componentIndex < c_virialAndEnergyCount);

        /* Reduce 7 outputs per warp in the shared memory */
        const int        maxBlockSize        = PME_SOLVE_THREADS_PER_BLOCK;
        const int        stride              = 8; // this is c_virialAndEnergyCount==7 rounded up to power of 2 for convenience
        const int        reductionBufferSize = (maxBlockSize / warp_size) * stride;
        __shared__ float sm_virialAndEnergy[reductionBufferSize];

        if (validComponentIndex)
        {
            const int warpIndex = threadLocalId / warp_size;
            sm_virialAndEnergy[warpIndex * stride + componentIndex] = virxx;
        }
        __syncthreads();

        /* Reduce to the single warp size */
        const int targetIndex = threadLocalId;
#pragma unroll
        for (int reductionStride = reductionBufferSize >> 1; reductionStride >= warp_size; reductionStride >>= 1)
        {
            if (targetIndex < reductionStride)
            {
                const int sourceIndex = targetIndex + reductionStride;
                sm_virialAndEnergy[targetIndex] += sm_virialAndEnergy[sourceIndex];
            }
            __syncthreads();
        }

        /* Now use shuffle again */
        if (threadLocalId < warp_size)
        {
            float output = sm_virialAndEnergy[threadLocalId];
#pragma unroll
            for (int delta = stride; delta < warp_size; delta <<= 1)
            {
                output += __shfl_down(output, delta, warp_size);
            }
            /* Final output */
            if (validComponentIndex)
            {
                atomicAdd(gm_virialAndEnergy + componentIndex, output);
            }
        }
#else
        /* A 7-thread energy and virial reduction in shared memory, inspired by reduce_force_j_generic */
        const int        maxBlockSize = PME_SOLVE_THREADS_PER_BLOCK;
        __shared__ float sm_virialAndEnergy[c_virialAndEnergyCount * maxBlockSize];
        sm_virialAndEnergy[threadLocalId + 0 * maxBlockSize] = virxx;
        sm_virialAndEnergy[threadLocalId + 1 * maxBlockSize] = viryy;
        sm_virialAndEnergy[threadLocalId + 2 * maxBlockSize] = virzz;
        sm_virialAndEnergy[threadLocalId + 3 * maxBlockSize] = virxy;
        sm_virialAndEnergy[threadLocalId + 4 * maxBlockSize] = virxz;
        sm_virialAndEnergy[threadLocalId + 5 * maxBlockSize] = viryz;
        sm_virialAndEnergy[threadLocalId + 6 * maxBlockSize] = energy;
        // zero the rest - is there a better way to do this?
        const int blockSize       = blockDim.x * blockDim.y * blockDim.z;
        const int paddingCount    = maxBlockSize - blockSize;
        if (threadLocalId < paddingCount)
        {
#pragma unroll
            for (int i = 0; i < c_virialAndEnergyCount; i++)
            {
                sm_virialAndEnergy[i * maxBlockSize + blockSize + threadLocalId] = 0;
            }
        }

        __syncthreads();

        /* Reducing every component to fit into warp_size */
        const int targetIndex = threadLocalId;
        for (int reductionStride = maxBlockSize >> 1; reductionStride >= warp_size; reductionStride >>= 1)
        {
            if (targetIndex < reductionStride)
            {
                const int sourceIndex = targetIndex + reductionStride;
#pragma unroll
                for (int i = 0; i < c_virialAndEnergyCount; i++)
                {
                    sm_virialAndEnergy[i * maxBlockSize + targetIndex] += sm_virialAndEnergy[i * maxBlockSize + sourceIndex];
                }
            }
            __syncthreads();
        }

        const int threadsPerComponent    = warp_size / c_virialAndEnergyCount; // this is also the stride, will be 32 / 7 = 4
        const int contributionsPerThread = warp_size / threadsPerComponent;    // will be 32 / 4 = 8
        if (threadLocalId < c_virialAndEnergyCount * threadsPerComponent)
        {
            const int componentIndex        = threadLocalId / threadsPerComponent;
            const int threadComponentOffset = threadLocalId - componentIndex * threadsPerComponent;

            float     sum = 0.0f;
#pragma unroll
            for (int j = 0; j < contributionsPerThread; j++)
            {
                sum += sm_virialAndEnergy[componentIndex * maxBlockSize + j * threadsPerComponent + threadComponentOffset];
            }
            atomicAdd(gm_virialAndEnergy + componentIndex, sum);
        }
#endif
    }
}

void pme_gpu_solve(const pme_gpu_t *pmeGpu, t_complex *h_grid,
                   bool computeEnergyAndVirial, GridOrdering gridOrdering)
{
    const bool   copyInputAndOutputGrid = pme_gpu_is_testing(pmeGpu) || !pme_gpu_performs_FFT(pmeGpu);

    cudaStream_t stream          = pmeGpu->archSpecific->pmeStream;
    const auto  *kernelParamsPtr = pmeGpu->kernelParams.get();

    if (copyInputAndOutputGrid)
    {
        cu_copy_H2D_async(kernelParamsPtr->grid.d_fourierGrid, h_grid, pmeGpu->archSpecific->complexGridSize * sizeof(float), stream);
    }

    int majorDim = -1, middleDim = -1, minorDim = -1;
    switch (gridOrdering)
    {
        case GridOrdering::YZX:
            majorDim  = YY;
            middleDim = ZZ;
            minorDim  = XX;
            break;

        case GridOrdering::XYZ:
            majorDim  = XX;
            middleDim = YY;
            minorDim  = ZZ;
            break;

        default:
            GMX_ASSERT(false, "Implement grid ordering here and below for the kernel launch");
    }

    const int   maxBlockSize      = PME_SOLVE_THREADS_PER_BLOCK;
    const int   gridLineSize      = pmeGpu->kernelParams->grid.complexGridSizePadded[minorDim];
    const int   gridLinesPerBlock = max(maxBlockSize / gridLineSize, 1);
    const int   blocksPerGridLine = (gridLineSize + maxBlockSize - 1) / maxBlockSize; // How many blocks would we need to process a single (large enough) gridline?
    dim3 threads(gridLineSize, gridLinesPerBlock);
    dim3 blocks(blocksPerGridLine,
                (pmeGpu->kernelParams->grid.complexGridSize[middleDim] + gridLinesPerBlock - 1) / gridLinesPerBlock,
                pmeGpu->kernelParams->grid.complexGridSize[majorDim]);

    pme_gpu_start_timing(pmeGpu, gtPME_SOLVE);
    if (gridOrdering == GridOrdering::YZX)
    {
        if (computeEnergyAndVirial)
        {
            pme_solve_kernel<true, GridOrderingInternal::YZX> <<< blocks, threads, 0, stream>>> (*kernelParamsPtr);
        }
        else
        {
            pme_solve_kernel<false, GridOrderingInternal::YZX> <<< blocks, threads, 0, stream>>> (*kernelParamsPtr);
        }
    }
    else if (gridOrdering == GridOrdering::XYZ)
    {
        if (computeEnergyAndVirial)
        {
            pme_solve_kernel<true, GridOrderingInternal::XYZ> <<< blocks, threads, 0, stream>>> (*kernelParamsPtr);
        }
        else
        {
            pme_solve_kernel<false, GridOrderingInternal::XYZ> <<< blocks, threads, 0, stream>>> (*kernelParamsPtr);
        }
    }
    CU_LAUNCH_ERR("pme_solve_kernel");
    pme_gpu_stop_timing(pmeGpu, gtPME_SOLVE);

    if (computeEnergyAndVirial)
    {
        cu_copy_D2H_async(pmeGpu->staging.h_virialAndEnergy, kernelParamsPtr->constants.d_virialAndEnergy,
                          c_virialAndEnergyCount * sizeof(float), stream);
        cudaError_t stat = cudaEventRecord(pmeGpu->archSpecific->syncEnerVirD2H, stream);
        CU_RET_ERR(stat, "PME solve energy/virial event record failure");
    }

    if (copyInputAndOutputGrid)
    {
        cu_copy_D2H_async(h_grid, kernelParamsPtr->grid.d_fourierGrid, pmeGpu->archSpecific->complexGridSize * sizeof(float), stream);
        cudaError_t stat = cudaEventRecord(pmeGpu->archSpecific->syncSolveGridD2H, stream);
        CU_RET_ERR(stat, "PME solve grid sync event record failure");
    }
}
