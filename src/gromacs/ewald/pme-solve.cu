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

#include "gromacs/utility/gmxassert.h"
#include "pme.cuh"
#include "pme-gpu.h"   //?
#include "pme-internal.h"
#include "pme-solve.h" //? some work structure reliance?

void pme_gpu_alloc_energy_virial(const gmx_pme_t *pme, const int gmx_unused grid_index)
{
    pme->gpu->energyAndVirialSize = 7 * sizeof(real); /* 6 virial components + energy */
    pme->gpu->energyAndVirial     = (real *)PMEMemoryFetch(pme, PME_ID_ENERGY_AND_VIRIAL, pme->gpu->energyAndVirialSize, ML_DEVICE);
}

void pme_gpu_clear_energy_virial(const gmx_pme_t *pme, const int gmx_unused grid_index)
{
    cudaError_t stat = cudaMemsetAsync(pme->gpu->energyAndVirial, 0, pme->gpu->energyAndVirialSize, pme->gpu->pmeStream);
    CU_RET_ERR(stat, "PME solve energies/virial cudaMemsetAsync");
}

/*! \brief
 *
 * Copies the pre-computed B-spline modules to the GPU
 */
void pme_gpu_copy_bspline_moduli(const gmx_pme_t *pme)
{
    for (int i = 0; i < DIM; i++)
    {
        int       gridSize;
        PMEDataID id;
        switch (i)
        {
            case XX:
                gridSize = pme->nkx;
                id       = PME_ID_BSP_MOD_XX;
                break;

            case YY:
                gridSize = pme->nky;
                id       = PME_ID_BSP_MOD_YY;
                break;

            case ZZ:
                gridSize = pme->nkz;
                id       = PME_ID_BSP_MOD_ZZ;
                break;
        }
        int   modSize  = gridSize * sizeof(real);
        real *bspMod_h = (real *)PMEMemoryFetch(pme, id, modSize, ML_HOST);
        memcpy(bspMod_h, pme->bsp_mod[i], modSize);
        real *bspMod_d = (real *)PMEMemoryFetch(pme, id, modSize, ML_DEVICE);
        cu_copy_H2D_async(bspMod_d, bspMod_h, modSize, pme->gpu->pmeStream);
    }
}


#define THREADS_PER_BLOCK (4 * warp_size)

template<
    const gmx_bool bEnerVir,
    // should the energy/virial be computed
    const gmx_bool YZXOrdering
    // false - GPU solve works in a XYZ ordering (after a single-rank cuFFT)
    // true - GPU solve works in a YZX ordering, like the CPU one (after FFTW)
    >
__global__ void pme_solve_kernel
    (const int localCountMajor, const int localCountMiddle, const int localCountMinor,
    const int localOffsetMinor, const int localOffsetMajor, const int localOffsetMiddle,
    const int localSizeMinor, /*const int localSizeMajor,*/ const int localSizeMiddle,
    const real * __restrict__ BSplineModuleMinor,
    const real * __restrict__ BSplineModuleMajor,
    const real * __restrict__ BSplineModuleMiddle,
    float2 * __restrict__ globalGrid,
    const struct pme_gpu_const_parameters constants,
    real * __restrict__ virialAndEnergy)
{
    // this is a PME solve kernel
    // each thread works on one cell of the Fourier space complex 3D grid (float2 * __restrict__ grid)
    // each block handles THREADS_PER_BLOCK cells - depending on the grid contiguous dimension size,
    // that can range from a part of a single gridline to several complete gridlines
    // the minor dimension index is (YZXOrdering ? XX : ZZ)
    const int threadLocalId = (threadIdx.y * blockDim.x) + threadIdx.x;
    //const int blockSize = (blockDim.x * blockDim.y * blockDim.z); // == cellsPerBlock
    const int blockSize = THREADS_PER_BLOCK;
    //const int threadId = blockId * blockSize + threadLocalId;

    const int nMinor  = !YZXOrdering ? constants.localGridSize.z : constants.localGridSize.x; //yupinov fix all pme->nkx and such
    const int nMajor  = !YZXOrdering ? constants.localGridSize.x : constants.localGridSize.y;
    const int nMiddle = !YZXOrdering ? constants.localGridSize.y : constants.localGridSize.z;

    int       maxkMajor  = (nMajor + 1) / 2;  //X or Y
    int       maxkMiddle = (nMiddle + 1) / 2; //Y OR Z => only check for !YZX
    int       maxkMinor  = (nMinor + 1) / 2;  //Z or X => only check for YZX

    const int enerVirSize = 7;

    real      energy = 0.0f;
    real      virxx  = 0.0f, virxy = 0.0f, virxz = 0.0f, viryy = 0.0f, viryz = 0.0f, virzz = 0.0f;

    const int indexMinor  = blockIdx.x * blockDim.x + threadIdx.x;
    const int indexMiddle = blockIdx.y * blockDim.y + threadIdx.y;
    const int indexMajor  = blockIdx.z * blockDim.z + threadIdx.z;

    if ((indexMajor < localCountMajor) && (indexMiddle < localCountMiddle) && (indexMinor < localCountMinor))
    {
        /* The offset should be equal to the global thread index */
        float2    *globalGridPtr = globalGrid + (indexMajor * localSizeMiddle + indexMiddle) * localSizeMinor + indexMinor;

        const int  kMajor = indexMajor + localOffsetMajor;
        /* Checking either X in XYZ, or Y in YZX cases */
        const real mMajor = (kMajor < maxkMajor) ? kMajor : (kMajor - nMajor);

        const int  kMiddle = indexMiddle + localOffsetMiddle;
        real       mMiddle = kMiddle;
        /* Checking Y in XYZ case */
        if (!YZXOrdering)
        {
            mMiddle = (kMiddle < maxkMiddle) ? kMiddle : (kMiddle - nMiddle);
        }
        /* We should skip the k-space point (0,0,0) */

        const int      kMinor       = localOffsetMinor + indexMinor;
        const gmx_bool notZeroPoint = (kMinor > 0 || kMajor > 0 || kMiddle > 0);
        real           mMinor       = kMinor, mhxk, mhyk, mhzk, m2k;

        /* Checking X in YZX case */
        if (YZXOrdering)
        {
            mMinor = (kMinor < maxkMinor) ? kMinor : (kMinor - nMinor);
        }

        real mX, mY, mZ;
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
        real corner_fac = 1.0f;
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
            mhxk       = mX * constants.recipbox[XX].x;
            mhyk       = mX * constants.recipbox[XX].y + mY * constants.recipbox[YY].y;
            mhzk       = mX * constants.recipbox[XX].z + mY * constants.recipbox[YY].z + mZ * constants.recipbox[ZZ].z;

            m2k        = mhxk * mhxk + mhyk * mhyk + mhzk * mhzk;
            real denom = m2k * real(M_PI) * constants.volume * BSplineModuleMajor[kMajor] * BSplineModuleMiddle[kMiddle] * BSplineModuleMinor[kMinor];
            real tmp1  = -constants.ewaldFactor * m2k;

            denom = 1.0f / denom;
            tmp1  = expf(tmp1);
            real   etermk = constants.elFactor * tmp1 * denom;

            float2 gridValue    = *globalGridPtr;
            float2 oldGridValue = gridValue;
            gridValue.x   *= etermk;
            gridValue.y   *= etermk;
            *globalGridPtr = gridValue;

            if (bEnerVir)
            {
                real tmp1k = 2.0f * (gridValue.x * oldGridValue.x + gridValue.y * oldGridValue.y);

                real vfactor = (constants.ewaldFactor + 1.0f / m2k) * 2.0f;
                real ets2    = corner_fac * tmp1k;
                energy = ets2;

                real ets2vf  = ets2 * vfactor;

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
            __shared__ real virialAndEnergyShared[enerVirSize * blockSize];
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
                for (int i = 0; i < enerVirSize; i++)
                {
                    if (threadLocalId < s) // split per threads?
                    {
                        virialAndEnergyShared[i * blockSize + threadLocalId] += virialAndEnergyShared[i * blockSize + threadLocalId + s];
                    }
                }
                __syncthreads();
            }

            const int threadsPerComponent    = warp_size / enerVirSize;         // this is also the stride, will be 32 / 7 = 4
            const int contributionsPerThread = warp_size / threadsPerComponent; // will be 32 / 4 = 8
            if (threadLocalId < enerVirSize * threadsPerComponent)
            {
                const int componentIndex        = threadLocalId / threadsPerComponent;
                const int threadComponentOffset = threadLocalId - componentIndex * threadsPerComponent;

                float     sum = 0.0f;
#pragma unroll
                for (int j = 0; j < contributionsPerThread; j++)
                {
                    sum += virialAndEnergyShared[componentIndex * blockSize + j * threadsPerComponent + threadComponentOffset];
                }
                // write to global memory
                atomicAdd(virialAndEnergy + componentIndex, sum);
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
                atomicAdd(virialAndEnergy + threadLocalId, virialAndEnergyShared[threadLocalId]);
               }
             */
        }
    }
}

void solve_pme_gpu(struct gmx_pme_t *pme, t_complex *grid,
                   gmx_bool bEnerVir)
{
    /* do recip sum over local cells in grid */

    const gmx_bool YZXOrdering = !pme->gpu->bGPUFFT;
    /* true: y major, z middle, x minor or continuous - the CPU FFTW way */
    /* false: x major, y middle, z minor - the single rank GPU cuFFT way */

    cudaStream_t s = pme->gpu->pmeStream;

    ivec         local_ndata, local_offset, local_size, complex_order;
    /* Dimensions should be identical for A/B grid, so we just use A here */

    if (pme->gpu->bGPUFFT)
    {
        gmx_parallel_3dfft_complex_limits_gpu(pme->gpu->pfft_setup_gpu[PME_GRID_QA], local_ndata, local_offset, local_size);
    }
    else
    {
        gmx_parallel_3dfft_complex_limits(pme->pfft_setup[PME_GRID_QA], complex_order, local_ndata, local_offset, local_size);
    }

    const int minorDim  = !YZXOrdering ? ZZ : XX;
    const int middleDim = !YZXOrdering ? YY : ZZ;
    const int majorDim  = !YZXOrdering ? XX : YY;

    /*
       const int nMinor =  local_ndata[minorDim]; //!YZXOrdering ? pme->nkz : pme->nkx;
       const int nMajor = local_ndata[majorDim];
       const int nMiddle = local_ndata[middleDim]; //these are basic sizes, so what
     */
    const real *bspModMinor_d  = (real *)PMEMemoryFetch(pme, !YZXOrdering ? PME_ID_BSP_MOD_ZZ : PME_ID_BSP_MOD_XX, 0, ML_DEVICE);
    const real *bspModMiddle_d = (real *)PMEMemoryFetch(pme, !YZXOrdering ? PME_ID_BSP_MOD_YY : PME_ID_BSP_MOD_ZZ, 0, ML_DEVICE);
    const real *bspModMajor_d  = (real *)PMEMemoryFetch(pme, !YZXOrdering ? PME_ID_BSP_MOD_XX : PME_ID_BSP_MOD_YY, 0, ML_DEVICE);

    const int   gridSize = local_size[XX] * local_size[YY] * local_size[ZZ] * sizeof(float2);

    float2     *grid_d = (float2 *)pme->gpu->fourierGrid;
    if (!pme->gpu->bGPUFFT)
    {
        cu_copy_H2D_async(grid_d, grid, gridSize, s);
    }

    // Z-dimension is too small in CUDA limitations (64 on CC30?), so instead of major-middle-minor sizing we do minor-middle-major
    const int maxBlockSize      = THREADS_PER_BLOCK;
    const int gridLineSize      = local_size[minorDim];
    const int gridLinesPerBlock = max(maxBlockSize / gridLineSize, 1);
    const int blocksPerGridLine = (gridLineSize + maxBlockSize - 1) / maxBlockSize; // rounded up
    dim3 threads((maxBlockSize + gridLinesPerBlock - 1) / gridLinesPerBlock, gridLinesPerBlock);
    const int blockSize = threads.x * threads.y * threads.z;
    GMX_RELEASE_ASSERT(blockSize >= maxBlockSize, "wrong PME GPU solve launch parameters");
    // we want to have spare threads to zero all the shared memory which we use in CC2.0 shared mem reduction

    dim3 blocks(blocksPerGridLine,
                (local_ndata[middleDim] + gridLinesPerBlock - 1) / gridLinesPerBlock, // rounded up middle dimension block number
                local_ndata[majorDim]);

    pme_gpu_timing_start(pme, ewcsPME_SOLVE);

    if (YZXOrdering)
    {
        if (bEnerVir)
        {
            pme_solve_kernel<TRUE, TRUE> <<< blocks, threads, 0, s>>>
            (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim],
             local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
             local_size[minorDim], /*local_size[majorDim],*/ local_size[middleDim],
             bspModMinor_d, bspModMajor_d, bspModMiddle_d,
             grid_d,
             pme->gpu->constants,
             pme->gpu->energyAndVirial);
        }
        else
        {
            pme_solve_kernel<FALSE, TRUE> <<< blocks, threads, 0, s>>>
            (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim ],
             local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
             local_size[minorDim], /*local_size[majorDim],*/ local_size[middleDim],
             bspModMinor_d, bspModMajor_d, bspModMiddle_d,
             grid_d,
             pme->gpu->constants,
             pme->gpu->energyAndVirial);
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
             bspModMinor_d, bspModMajor_d, bspModMiddle_d,
             grid_d,
             pme->gpu->constants,
             pme->gpu->energyAndVirial);
        }
        else
        {
            pme_solve_kernel<FALSE, FALSE> <<< blocks, threads, 0, s>>>
            (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim ],
             local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
             local_size[minorDim], /*local_size[majorDim],*/ local_size[middleDim],
             bspModMinor_d, bspModMajor_d, bspModMiddle_d,
             grid_d,
             pme->gpu->constants,
             pme->gpu->energyAndVirial);
        }
    }
    CU_LAUNCH_ERR("pme_solve_kernel");

    pme_gpu_timing_stop(pme, ewcsPME_SOLVE);

    if (bEnerVir)
    {
        real       *energyAndVirial_h = (real *)PMEMemoryFetch(pme, PME_ID_ENERGY_AND_VIRIAL, pme->gpu->energyAndVirialSize, ML_HOST);
        cu_copy_D2H_async(energyAndVirial_h, pme->gpu->energyAndVirial, pme->gpu->energyAndVirialSize, s);
        cudaError_t stat = cudaEventRecord(pme->gpu->syncEnerVirD2H, s);
        CU_RET_ERR(stat, "PME solve energy/virial sync fail");
    }

    if (!pme->gpu->bGPUFFT)
    {
        cu_copy_D2H_async(grid, grid_d, gridSize, s);
        cudaError_t stat = cudaEventRecord(pme->gpu->syncSolveGridD2H, s);
        CU_RET_ERR(stat, "PME solve grid sync fail");
    }
}

void pme_gpu_get_energy_virial(const gmx_pme_t *pme)
{
    cudaStream_t             s = pme->gpu->pmeStream;

    struct pme_solve_work_t *work          = &pme->solve_work[0];
    real                    *work_energy_q = &(work->energy_q);
    matrix                  &work_vir_q    = work->vir_q;

    cudaError_t              stat = cudaStreamWaitEvent(s, pme->gpu->syncEnerVirD2H, 0);
    CU_RET_ERR(stat, "error while waiting for PME solve");
    real                    *energyAndVirial_h = (real *)PMEMemoryFetch(pme, PME_ID_ENERGY_AND_VIRIAL, pme->gpu->energyAndVirialSize, ML_HOST);
    real                     energy            = 0.0;
    real                     virxx             = 0.0, virxy = 0.0, virxz = 0.0, viryy = 0.0, viryz = 0.0, virzz = 0.0;

    int j = 0;
    virxx  += energyAndVirial_h[j++];
    viryy  += energyAndVirial_h[j++];
    virzz  += energyAndVirial_h[j++];
    virxy  += energyAndVirial_h[j++];
    virxz  += energyAndVirial_h[j++];
    viryz  += energyAndVirial_h[j++];
    energy += energyAndVirial_h[j++];
    for (j = 0; j < 7; j++)
    {
        GMX_RELEASE_ASSERT(!isnan(energyAndVirial_h[j]), "PME GPU is broken - NaN reduction result");
    }

    work_vir_q[XX][XX] = 0.25 * virxx;
    work_vir_q[YY][YY] = 0.25 * viryy;
    work_vir_q[ZZ][ZZ] = 0.25 * virzz;
    work_vir_q[XX][YY] = work_vir_q[YY][XX] = 0.25 * virxy;
    work_vir_q[XX][ZZ] = work_vir_q[ZZ][XX] = 0.25 * virxz;
    work_vir_q[YY][ZZ] = work_vir_q[ZZ][YY] = 0.25 * viryz;

    /* This energy should be corrected for a charged system */
    *work_energy_q = 0.5 * energy;
}
