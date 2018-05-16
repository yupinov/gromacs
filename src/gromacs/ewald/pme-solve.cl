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
#include "pme-gpu-types.h"

/*! \brief
 * PME complex grid solver kernel function.
 *
 * \tparam[in] gridOrdering             Specifies the dimension ordering of the complex grid.
 * \tparam[in] computeEnergyAndVirial   Tells if the reciprocal energy and virial should be computed.
 * \param[in]  kernelParams             Input PME CUDA data in constant memory. FIXME docs
 */
__kernel void CUSTOMIZED_KERNEL_NAME(pme_solve_kernel)(const struct PmeGpuCudaKernelParams kernelParams,
                                                       GLOBAL const float * __restrict__   gm_splineModuli,
                                                       GLOBAL float * __restrict__         gm_virialAndEnergy,
                                                       GLOBAL float2 * __restrict__        gm_grid
                                                       )
{
    /* This kernel supports 2 different grid dimension orderings: YZX and XYZ */
    int majorDim, middleDim, minorDim;
    if (gridOrdering == YZX)
    {
        majorDim  = YY;
        middleDim = ZZ;
        minorDim  = XX;
    }
    if (gridOrdering == XYZ)
    {
        majorDim  = XX;
        middleDim = YY;
        minorDim  = ZZ;
    }

    GLOBAL const float * __restrict__ gm_splineValueMajor   = gm_splineModuli + kernelParams.grid.splineValuesOffset[majorDim];
    GLOBAL const float * __restrict__ gm_splineValueMiddle  = gm_splineModuli + kernelParams.grid.splineValuesOffset[middleDim];
    GLOBAL const float * __restrict__ gm_splineValueMinor   = gm_splineModuli + kernelParams.grid.splineValuesOffset[minorDim];

    /* Various grid sizes and indices */
    const int localOffsetMinor = 0, localOffsetMajor = 0, localOffsetMiddle = 0; //unused
    const int localSizeMinor   = kernelParams.grid.complexGridSizePadded[minorDim];
    const int localSizeMiddle  = kernelParams.grid.complexGridSizePadded[middleDim];
    const int localCountMiddle = kernelParams.grid.complexGridSize[middleDim];
    const int localCountMinor  = kernelParams.grid.complexGridSize[minorDim];
    const int nMajor           = kernelParams.grid.realGridSize[majorDim];
    const int nMiddle          = kernelParams.grid.realGridSize[middleDim];
    const int nMinor           = kernelParams.grid.realGridSize[minorDim];
    const int maxkMajor        = (nMajor + 1) / 2;  // X or Y
    const int maxkMiddle       = (nMiddle + 1) / 2; // Y OR Z => only check for !YZX
    const int maxkMinor        = (nMinor + 1) / 2;  // Z or X => only check for YZX

    /* Each thread works on one cell of the Fourier space complex 3D grid (gm_grid).
     * Each block handles up to c_solveMaxThreadsPerBlock cells -
     * depending on the grid contiguous dimension size,
     * that can range from a part of a single gridline to several complete gridlines.
     */
    const int threadLocalId     = getThreadLocalIndex(XX);
    const int gridLineSize      = localCountMinor;
    const int gridLineIndex     = threadLocalId / gridLineSize;
    const int gridLineCellIndex = threadLocalId - gridLineSize * gridLineIndex;
    const int gridLinesPerBlock = getBlockSize(XX) / gridLineSize;
    const int activeWarps       = (getBlockSize(XX) / warp_size);
    const int indexMinor        = getBlockIndex(XX) * getBlockSize(XX) + gridLineCellIndex;
    const int indexMiddle       = getBlockIndex(YY) * gridLinesPerBlock + gridLineIndex;
    const int indexMajor        = getBlockIndex(ZZ);

    /* Optional outputs */
    float energy = 0.0f;
    float virxx  = 0.0f;
    float virxy  = 0.0f;
    float virxz  = 0.0f;
    float viryy  = 0.0f;
    float viryz  = 0.0f;
    float virzz  = 0.0f;

    assert(indexMajor < kernelParams.grid.complexGridSize[majorDim]);
    if ((indexMiddle < localCountMiddle) & (indexMinor < localCountMinor) & (gridLineIndex < gridLinesPerBlock))
    {
        /* The offset should be equal to the global thread index for coalesced access */
        const int                    gridIndex     = (indexMajor * localSizeMiddle + indexMiddle) * localSizeMinor + indexMinor;
        GLOBAL float2 * __restrict__ gm_gridCell   = gm_grid + gridIndex;

        const int                    kMajor  = indexMajor + localOffsetMajor;
        /* Checking either X in XYZ, or Y in YZX cases */
        const float                  mMajor  = (kMajor < maxkMajor) ? kMajor : (kMajor - nMajor);

        const int                    kMiddle = indexMiddle + localOffsetMiddle;
        float                        mMiddle = kMiddle;
        /* Checking Y in XYZ case */
        if (gridOrdering == XYZ)
        {
            mMiddle = (kMiddle < maxkMiddle) ? kMiddle : (kMiddle - nMiddle);
        }
        const int             kMinor  = localOffsetMinor + indexMinor;
        float                 mMinor  = kMinor;
        /* Checking X in YZX case */
        if (gridOrdering == YZX)
        {
            mMinor = (kMinor < maxkMinor) ? kMinor : (kMinor - nMinor);
        }
        /* We should skip the k-space point (0,0,0) */
        const bool notZeroPoint  = (kMinor > 0) | (kMajor > 0) | (kMiddle > 0);

        float      mX, mY, mZ;
        if (gridOrdering == YZX)
        {
            mX = mMinor;
            mY = mMajor;
            mZ = mMiddle;
        }
        if (gridOrdering == XYZ)
        {
            mX = mMajor;
            mY = mMiddle;
            mZ = mMinor;
        }

        /* 0.5 correction factor for the first and last components of a Z dimension */
        float corner_fac = 1.0f;
        if (gridOrdering == YZX)
        {
            if ((kMiddle == 0) | (kMiddle == maxkMiddle))
            {
                corner_fac = 0.5f;
            }
        }
        if (gridOrdering == XYZ)
        {
            if ((kMinor == 0) | (kMinor == maxkMinor))
            {
                corner_fac = 0.5f;
            }
        }

        if (notZeroPoint)
        {
            const float mhxk = mX * kernelParams.current.recipBox[XX][XX];
            const float mhyk = mX * kernelParams.current.recipBox[XX][YY] + mY * kernelParams.current.recipBox[YY][YY];
            const float mhzk = mX * kernelParams.current.recipBox[XX][ZZ] + mY * kernelParams.current.recipBox[YY][ZZ] + mZ * kernelParams.current.recipBox[ZZ][ZZ];

            const float m2k        = mhxk * mhxk + mhyk * mhyk + mhzk * mhzk;
            assert(m2k != 0.0f);
            //TODO: use LDG/textures for gm_splineValue
            float       denom = m2k * M_PI_F * kernelParams.current.boxVolume * gm_splineValueMajor[kMajor] * gm_splineValueMiddle[kMiddle] * gm_splineValueMinor[kMinor];
            assert(isfinite(denom));
            assert(denom != 0.0f);
            const float   tmp1   = exp(-kernelParams.grid.ewaldFactor * m2k); //FIXME was expf in CUDA
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

    // this is only for reduction
    SHARED float sm_virialAndEnergy[c_virialAndEnergyCount * warp_size];

    /* Optional energy/virial reduction */
    if (computeEnergyAndVirial)
    {
        // TODO: implement AMD intrinsics reduction, like with shuffles in CUDA version.

        /* Shared memory reduction with atomics.
         * Each component is first reduced into warp_size positions in the shared memory;
         * Then first c_virialAndEnergyCount warps reduce everything further and add to the global memory.
         * This can likely be improved, but is anyway faster than the previous straightforward reduction,
         * which was using too much shared memory (for storing all 7 floats on each thread).
         */

        const int  lane      = threadLocalId & (warp_size - 1);
        const int  warpIndex = threadLocalId / warp_size;
        const bool firstWarp = (warpIndex == 0);
        if (firstWarp)
        {
            sm_virialAndEnergy[0 * warp_size + lane] = virxx;
            sm_virialAndEnergy[1 * warp_size + lane] = viryy;
            sm_virialAndEnergy[2 * warp_size + lane] = virzz;
            sm_virialAndEnergy[3 * warp_size + lane] = virxy;
            sm_virialAndEnergy[4 * warp_size + lane] = virxz;
            sm_virialAndEnergy[5 * warp_size + lane] = viryz;
            sm_virialAndEnergy[6 * warp_size + lane] = energy;
        }
        sharedMemoryBarrier();
        if (!firstWarp)
        {
            atomicAdd_l_f(sm_virialAndEnergy + 0 * warp_size + lane, virxx);
            atomicAdd_l_f(sm_virialAndEnergy + 1 * warp_size + lane, viryy);
            atomicAdd_l_f(sm_virialAndEnergy + 2 * warp_size + lane, virzz);
            atomicAdd_l_f(sm_virialAndEnergy + 3 * warp_size + lane, virxy);
            atomicAdd_l_f(sm_virialAndEnergy + 4 * warp_size + lane, virxz);
            atomicAdd_l_f(sm_virialAndEnergy + 5 * warp_size + lane, viryz);
            atomicAdd_l_f(sm_virialAndEnergy + 6 * warp_size + lane, energy);
        }
        sharedMemoryBarrier();

        const int numIter = (c_virialAndEnergyCount + activeWarps - 1) / activeWarps;
        for (int i = 0; i < numIter; i++)
        {
            const int componentIndex = i * activeWarps + warpIndex;
            if (componentIndex < c_virialAndEnergyCount)
            {
                const int targetIndex = componentIndex * warp_size + lane;
    #pragma unroll
                for (int reductionStride = warp_size >> 1; reductionStride >= 1; reductionStride >>= 1)
                {
                    if (lane < reductionStride)
                    {
                        sm_virialAndEnergy[targetIndex] += sm_virialAndEnergy[targetIndex + reductionStride];
                        //FIXME thsi added barrrier makes OpenCL correct
                        sharedMemoryBarrier();
                    }
                }
                if (lane == 0)
                {
                    atomicAdd(gm_virialAndEnergy + componentIndex, sm_virialAndEnergy[targetIndex]);
                }
            }
        }
    }
}
