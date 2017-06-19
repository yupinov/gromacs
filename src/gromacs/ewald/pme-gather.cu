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
 *  \brief Implements PME force gathering in CUDA.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include <cassert>

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"

#include "pme.cuh"
#include "pme-timings.cuh"

//! Gathering max block size
constexpr int c_gatherMaxThreadsPerBlock = (4 * warp_size);
//! Gathering min blocks per CUDA multiprocessor - for CC2.x, we just take the CUDA limit of 8 to avoid the warning
constexpr int c_gatherMinBlocksPerMP = (GMX_PTX_ARCH < 300) ? 8 : (GMX_CUDA_MAX_THREADS_PER_MP / c_gatherMaxThreadsPerBlock);

/*! \brief
 * An inline CUDA function: unroll the dynamic index accesses to the constant grid sizes to avoid local memory operations.
 */
__device__ __forceinline__ float read_grid_size(const float *realGridSizeFP,
                                                const int    dimIndex)
{
    switch (dimIndex)
    {
        case XX: return realGridSizeFP[XX];
        case YY: return realGridSizeFP[YY];
        case ZZ: return realGridSizeFP[ZZ];
    }
    assert(false);
    return 0.0f;
}

/*! \brief Reduce the order^2 contributions.
 *
 *  \param[out]    sm_forces          Shared memory array with the output forces (number of elements is number of atoms per block)
 *  \param[in]     atomIndexLocal     Local atom index
 *  \param[in]     splineIndex        Spline index
 *  \param[in]     lineIndex          Line index (same as threadLocalId)
 *  \param[in]     realGridSizeFP     Local grid size constant
 *  \param[in]     fx                 Input force partial component X
 *  \param[in]     fy                 Input force partial component Y
 *  \param[in]     fz                 Input force partial component Z
 *
 */
template <
    const int order,
    const int atomDataSize,
    const int blockSize
    >
__device__ __forceinline__ void reduce_atom_forces(float3 * __restrict__ sm_forces,
                                                   const int             atomIndexLocal,
                                                   const int             splineIndex,
                                                   const int             lineIndex,
                                                   const float          *realGridSizeFP,
                                                   float                &fx,
                                                   float                &fy,
                                                   float                &fz)
{
#if (GMX_PTX_ARCH >= 300)
    if (!(order & (order - 1))) // Only for orders of power of 2
    {
        // A tricky shuffle reduction inspired by reduce_force_j_warp_shfl
        static_assert(order == 4, "Only order of 4 is implemented");
        static_assert(atomDataSize <= warp_size, "TODO: rework for atomDataSize > warp_size (order 8 or larger)");
        const int width = atomDataSize;

        fx += __shfl_down(fx, 1, width);
        fy += __shfl_up  (fy, 1, width);
        fz += __shfl_down(fz, 1, width);

        if (splineIndex & 1)
        {
            fx = fy;
        }

        fx += __shfl_down(fx, 2, width);
        fz += __shfl_up  (fz, 2, width);

        if (splineIndex & 2)
        {
            fx = fz;
        }

        // By now fx contains intermediate quad sums of all 3 components:
        // splineIndex    0            1            2 and 3      4            5            6 and 7      8...
        // sum of...      fx0 to fx3   fy0 to fy3   fz0 to fz3   fx4 to fx7   fy4 to fy7   fz4 to fz7   etc.

        // We have to just further reduce those groups of 4
        for (int delta = 4; delta < atomDataSize; delta <<= 1)
        {
            fx += __shfl_down(fx, delta, width);
        }

        const int dimIndex = splineIndex;
        if (dimIndex < DIM)
        {
            const float n = read_grid_size(realGridSizeFP, dimIndex);
            *((float *)(&sm_forces[atomIndexLocal]) + dimIndex) = fx * n;
        }
    }
    else
#endif
    {
        // We use blockSize shared memory elements to read fx, or fy, or fz, and then reduce them to fit into smemPerDim elements
        // which are stored separately (first 2 dimensions only)
        const int         smemPerDim   = warp_size;
        const int         smemReserved = (DIM - 1) * smemPerDim;
        __shared__ float  sm_forceReduction[smemReserved + blockSize];
        __shared__ float *sm_forceTemp[DIM];

        const int         numWarps  = blockSize / smemPerDim;
        const int         minStride = max(1, atomDataSize / numWarps); // order 4: 128 threads => 4, 256 threads => 2, etc

#pragma unroll
        for (int dimIndex = 0; dimIndex < DIM; dimIndex++)
        {
            int elementIndex = smemReserved + lineIndex;
            // Store input force contributions
            sm_forceReduction[elementIndex] = (dimIndex == XX) ? fx : (dimIndex == YY) ? fy : fz;
            // Reduce to fit into smemPerDim (warp size)
#pragma unroll
            for (int redStride = atomDataSize / 2; redStride > minStride; redStride >>= 1)
            {
                if (splineIndex < redStride)
                {
                    sm_forceReduction[elementIndex] += sm_forceReduction[elementIndex + redStride];
                }
            }
            // Last iteration - packing everything to be nearby, storing convenience pointer
            sm_forceTemp[dimIndex] = sm_forceReduction + dimIndex * smemPerDim;
            int redStride = minStride;
            if (splineIndex < redStride)
            {
                const int packedIndex = atomIndexLocal * redStride + splineIndex;
                sm_forceTemp[dimIndex][packedIndex] = sm_forceReduction[elementIndex] + sm_forceReduction[elementIndex + redStride];
            }
        }

        __syncthreads();

        assert ((blockSize / warp_size) >= DIM);
        //assert (atomsPerBlock <= warp_size);

        const int warpIndex = lineIndex / warp_size;
        const int dimIndex  = warpIndex;

        // First 3 warps can now process 1 dimension each
        if (dimIndex < DIM)
        {
            int sourceIndex = lineIndex % warp_size; //?
#pragma unroll
            for (int redStride = minStride / 2; redStride > 1; redStride >>= 1)
            {
                if (!(splineIndex & redStride))
                {
                    sm_forceTemp[dimIndex][sourceIndex] += sm_forceTemp[dimIndex][sourceIndex + redStride];
                }
            }

            const float n = read_grid_size(realGridSizeFP, dimIndex);

            const int   atomIndex = sourceIndex / minStride;
            if (sourceIndex == minStride * atomIndex)
            {
                *((float *)(&sm_forces[atomIndex]) + dimIndex) = (sm_forceTemp[dimIndex][sourceIndex] + sm_forceTemp[dimIndex][sourceIndex + 1]) * n;
            }
        }

#if 0   // Previous lazy 3-thread reduction in shared memory inspired by reduce_force_j_generic
        __shared__ float sm_forceReduction[DIM * blockSize];
        sm_forceReduction[XX * blockSize + lineIndex] = fx;
        sm_forceReduction[YY * blockSize + lineIndex] = fy;
        sm_forceReduction[ZZ * blockSize + lineIndex] = fz;

        if (splineIndex < DIM) // This runs 6 threads (0..2, 16..18) in each warp
        {
            const float n = read_grid_size(realGridSizeFP, splineIndex);
            float       f = 0.0f;
            // This runs o^2==16 iterations over all single atom, single dimension contributions
            for (int j = atomIndexLocal * atomDataSize; j < (atomIndexLocal + 1) * atomDataSize; j++)
            {
                // Bank conflicts happen here!
                f += sm_forceReduction[blockSize * splineIndex + j];
            }
            *((float *)(&sm_forces[atomIndexLocal]) + splineIndex) = f * n;
        }
#endif
    }
}

/*! \brief
 * A CUDA kernel which gathers the atom forces from the grid.
 * The grid is assumed to be wrapped in dimension Z.
 *
 * \tparam[in] order                The PME order (must be 4).
 * \tparam[in] overwriteForces      True: the forces are written to the output buffer;
 *                                  False: the forces are added non-atomically to the output buffer (e.g. to the bonded forces).
 * \tparam[in] wrapX                Tells if the grid is wrapped in the X dimension.
 * \tparam[in] wrapY                Tells if the grid is wrapped in the Y dimension.
 * \param[in]  kernelParams         All the PME GPU data.
 */
template <
    const int order,
    const bool overwriteForces,
    const bool wrapX,
    const bool wrapY
    >
__launch_bounds__(c_gatherMaxThreadsPerBlock, c_gatherMinBlocksPerMP)
__global__ void pme_gather_kernel(const pme_gpu_cuda_kernel_params_t    kernelParams)
{
    /* Global memory pointers */
    const float * __restrict__  gm_coefficients     = kernelParams.atoms.d_coefficients;
    const float * __restrict__  gm_grid             = kernelParams.grid.d_realGrid;
    const float * __restrict__  gm_theta            = kernelParams.atoms.d_theta;
    const float * __restrict__  gm_dtheta           = kernelParams.atoms.d_dtheta;
    const int * __restrict__    gm_gridlineIndices  = kernelParams.atoms.d_gridlineIndices;
    float * __restrict__        gm_forces           = kernelParams.atoms.d_forces;

    /* Some sizes */
    const int    atomsPerBlock  = (c_gatherMaxThreadsPerBlock / PME_SPREADGATHER_THREADS_PER_ATOM);
    const int    atomDataSize   = PME_SPREADGATHER_THREADS_PER_ATOM; /* Number of data components and threads for a single atom */
    const int    blockSize      = atomsPerBlock * atomDataSize;

    /* These are the atom indices - for the shared and global memory */
    const int                   atomIndexLocal    = threadIdx.z;
    const int                   atomIndexOffset   = blockIdx.x * atomsPerBlock;
    const int                   atomIndexGlobal   = atomIndexOffset + atomIndexLocal;

    const size_t                splineParamsSize             = atomsPerBlock * DIM * order;
    const size_t                gridlineIndicesSize          = atomsPerBlock * DIM;
    __shared__ int              sm_gridlineIndices[gridlineIndicesSize];
    __shared__ float2           sm_splineParams[splineParamsSize]; /* Theta/dtheta pairs  as .x/.y */

    /* Spline Y/Z coordinates */
    const int ithy = threadIdx.y;
    const int ithz = threadIdx.x;

    /* These are the spline contribution indices in shared memory */
    const int splineIndex = threadIdx.y * blockDim.x + threadIdx.x;                  /* Relative to the current particle , 0..15 for order 4 */
    const int lineIndex   = (threadIdx.z * (blockDim.x * blockDim.y)) + splineIndex; /* And to all the block's particles */

    int       threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;

    /* Staging the atom gridline indices, DIM * atomsPerBlock threads */
    const int localGridlineIndicesIndex  = threadLocalId;
    const int globalGridlineIndicesIndex = blockIdx.x * gridlineIndicesSize + localGridlineIndicesIndex;
    const int globalCheckIndices         = pme_gpu_check_atom_data_index(globalGridlineIndicesIndex, kernelParams.atoms.nAtoms * DIM);
    if ((localGridlineIndicesIndex < gridlineIndicesSize) & globalCheckIndices)
    {
        sm_gridlineIndices[localGridlineIndicesIndex] = gm_gridlineIndices[globalGridlineIndicesIndex];
        assert(sm_gridlineIndices[localGridlineIndicesIndex] >= 0);
    }
    /* Staging the spline parameters, DIM * order * atomsPerBlock threads */
    const int localSplineParamsIndex  = threadLocalId;
    const int globalSplineParamsIndex = blockIdx.x * splineParamsSize + localSplineParamsIndex;
    const int globalCheckSplineParams = pme_gpu_check_atom_data_index(globalSplineParamsIndex, kernelParams.atoms.nAtoms * DIM * order);
    if ((localSplineParamsIndex < splineParamsSize) && globalCheckSplineParams)
    {
        sm_splineParams[localSplineParamsIndex].x = gm_theta[globalSplineParamsIndex];
        sm_splineParams[localSplineParamsIndex].y = gm_dtheta[globalSplineParamsIndex];
        assert(!isnan(sm_splineParams[localSplineParamsIndex].x));
        assert(!isnan(sm_splineParams[localSplineParamsIndex].y));
    }
    __syncthreads();

    float           fx = 0.0f;
    float           fy = 0.0f;
    float           fz = 0.0f;

    const int       globalCheck = pme_gpu_check_atom_data_index(atomIndexGlobal, kernelParams.atoms.nAtoms);
    const int       chargeCheck = pme_gpu_check_atom_charge(gm_coefficients[atomIndexGlobal]);

    if (chargeCheck & globalCheck)
    {
        const int    nx        = kernelParams.grid.realGridSize[XX];
        const int    ny        = kernelParams.grid.realGridSize[YY];
        const int    nz        = kernelParams.grid.realGridSize[ZZ];
        const int    pny       = kernelParams.grid.realGridSizePadded[YY];
        const int    pnz       = kernelParams.grid.realGridSizePadded[ZZ];

        const int    particleWarpIndex = atomIndexLocal % PME_SPREADGATHER_ATOMS_PER_WARP;
        const int    warpIndex         = atomIndexLocal / PME_SPREADGATHER_ATOMS_PER_WARP;

        const int    thetaOffsetBase = PME_SPLINE_THETA_STRIDE * order * warpIndex * DIM * PME_SPREADGATHER_ATOMS_PER_WARP + particleWarpIndex;
        const int    orderStride     = PME_SPLINE_THETA_STRIDE * DIM * PME_SPREADGATHER_ATOMS_PER_WARP;
        const int    dimStride       = PME_SPLINE_THETA_STRIDE * PME_SPREADGATHER_ATOMS_PER_WARP;

        const int    thetaOffsetY = thetaOffsetBase + ithy * orderStride + YY * dimStride;
        const float2 tdy          = sm_splineParams[thetaOffsetY];
        const int    thetaOffsetZ = thetaOffsetBase + ithz * orderStride + ZZ * dimStride;
        const float2 tdz          = sm_splineParams[thetaOffsetZ];

        const int    ixBase         = sm_gridlineIndices[atomIndexLocal * DIM + XX];
        int          iy             = sm_gridlineIndices[atomIndexLocal * DIM + YY] + ithy;
        if (wrapY & (iy >= ny))
        {
            iy -= ny;
        }
        int iz  = sm_gridlineIndices[atomIndexLocal * DIM + ZZ] + ithz;
        if (iz >= nz)
        {
            iz -= nz;
        }
        const int constOffset    = iy * pnz + iz;

#pragma unroll
        for (int ithx = 0; (ithx < order); ithx++)
        {
            int ix = ixBase + ithx;
            if (wrapX & (ix >= nx))
            {
                ix -= nx;
            }
            const int     gridIndexGlobal  = ix * pny * pnz + constOffset;
            assert(gridIndexGlobal >= 0);
            const float   gridValue    = gm_grid[gridIndexGlobal];
            assert(!isnan(gridValue));
            const int     thetaOffsetX = thetaOffsetBase + ithx * orderStride + XX * dimStride;
            const float2  tdx          = sm_splineParams[thetaOffsetX];
            const float   fxy1         = tdz.x * gridValue;
            const float   fz1          = tdz.y * gridValue;
            fx += tdx.y * tdy.x * fxy1;
            fy += tdx.x * tdy.y * fxy1;
            fz += tdx.x * tdy.x * fz1;
        }
    }
    //__syncthreads(); //not needed

    // Reduction of partial force contributions
    __shared__ float3 sm_forces[atomsPerBlock];
    reduce_atom_forces<order, atomDataSize, blockSize>(sm_forces,
                                                       atomIndexLocal, splineIndex, lineIndex,
                                                       kernelParams.grid.realGridSizeFP,
                                                       fx, fy, fz);
    __syncthreads();

    /* Calculating the final forces with no component branching, atomsPerBlock threads */
    const int forceIndexLocal  = threadLocalId;
    const int forceIndexGlobal = atomIndexOffset + forceIndexLocal;
    const int calcIndexCheck   = pme_gpu_check_atom_data_index(forceIndexGlobal, kernelParams.atoms.nAtoms);
    if ((forceIndexLocal < atomsPerBlock) & calcIndexCheck)
    {
        const float3  atomForces               = sm_forces[forceIndexLocal];
        const float   negCoefficient           = -gm_coefficients[forceIndexGlobal];
        float3        result;
        result.x                   = negCoefficient * kernelParams.step.recipBox[XX][XX] * atomForces.x;
        result.y                   = negCoefficient * (kernelParams.step.recipBox[XX][YY] * atomForces.x + kernelParams.step.recipBox[YY][YY] * atomForces.y);
        result.z                   = negCoefficient * (kernelParams.step.recipBox[XX][ZZ] * atomForces.x + kernelParams.step.recipBox[YY][ZZ] * atomForces.y + kernelParams.step.recipBox[ZZ][ZZ] * atomForces.z);
        sm_forces[forceIndexLocal] = result;
    }

    // No sync here
    assert(atomsPerBlock <= warp_size);

    /* Writing or adding the final forces component-wise, single warp */
    const size_t blockForcesSize = atomsPerBlock * DIM;
    const int    numIter         = (blockForcesSize + warp_size - 1) / warp_size;
    const int    iterThreads     = blockForcesSize / numIter;
    if (threadLocalId < iterThreads)
    {
#pragma unroll
        for (int i = 0; i < numIter; i++)
        {
            int         outputIndexLocal  = i * iterThreads + threadLocalId;
            int         outputIndexGlobal = blockIdx.x * blockForcesSize + outputIndexLocal;
            const int   globalOutputCheck = pme_gpu_check_atom_data_index(outputIndexGlobal, kernelParams.atoms.nAtoms * DIM);
            if (globalOutputCheck)
            {
                const float outputForceComponent = ((float *)sm_forces)[outputIndexLocal];
                if (overwriteForces)
                {
                    gm_forces[outputIndexGlobal] = outputForceComponent;
                }
                else
                {
                    gm_forces[outputIndexGlobal] += outputForceComponent;
                }
            }
        }
    }
}

void pme_gpu_gather(const pme_gpu_t *pmeGpu,
                    float           *h_forces,
                    bool             overwriteForces,
                    const float     *h_grid
                    )
{
    /* Copying the input CPU forces for reduction */
    if (!overwriteForces)
    {
        pme_gpu_copy_input_forces(pmeGpu, h_forces);
    }

    cudaStream_t stream          = pmeGpu->archSpecific->pmeStream;
    const int    order           = pmeGpu->common->pme_order;
    const auto  *kernelParamsPtr = pmeGpu->kernelParams.get();

    if (!pme_gpu_performs_FFT(pmeGpu) || pme_gpu_is_testing(pmeGpu))
    {
        pme_gpu_copy_input_gather_grid(pmeGpu, const_cast<float *>(h_grid));
    }

    if (pme_gpu_is_testing(pmeGpu))
    {
        /* Copying per-atom spline parameters */
        const size_t gridlineIndicesSizePerAtom = DIM * sizeof(int);
        const size_t splineDataSizePerAtom      = order * DIM * sizeof(float);
        const int    alignment                  = pme_gpu_get_atom_spline_data_alignment(pmeGpu);
        const size_t nAtomsSplineData           = ((pmeGpu->nAtomsAlloc + alignment - 1) / alignment) * alignment;
        if (c_usePadding)
        {
            // TODO: could clear only the padding and not the whole thing, but this is a test-exclusive code anyway
            CU_RET_ERR(cudaMemsetAsync(kernelParamsPtr->atoms.d_gridlineIndices, 0, pmeGpu->nAtomsAlloc * gridlineIndicesSizePerAtom, stream),
                       "PME failed to clear the gridline indices");
            CU_RET_ERR(cudaMemsetAsync(kernelParamsPtr->atoms.d_dtheta, 0, pmeGpu->nAtomsAlloc * splineDataSizePerAtom, stream),
                       "PME failed to clear the spline derivatives");
            CU_RET_ERR(cudaMemsetAsync(kernelParamsPtr->atoms.d_theta, 0, pmeGpu->nAtomsAlloc * splineDataSizePerAtom, stream),
                       "PME failed to clear the spline values");
        }
        cu_copy_H2D_async(kernelParamsPtr->atoms.d_gridlineIndices, pmeGpu->staging.h_gridlineIndices, kernelParamsPtr->atoms.nAtoms * gridlineIndicesSizePerAtom, stream);
        cu_copy_H2D_async(kernelParamsPtr->atoms.d_dtheta, pmeGpu->staging.h_dtheta, nAtomsSplineData * splineDataSizePerAtom, stream);
        cu_copy_H2D_async(kernelParamsPtr->atoms.d_theta, pmeGpu->staging.h_theta, nAtomsSplineData * splineDataSizePerAtom, stream);
    }

    const int atomsPerBlock  =  (c_gatherMaxThreadsPerBlock / PME_SPREADGATHER_THREADS_PER_ATOM);
    GMX_ASSERT(!c_usePadding || !(PME_ATOM_DATA_ALIGNMENT % atomsPerBlock), "inconsistent atom data padding vs. gathering block size");

    dim3 nBlocks(pmeGpu->nAtomsPadded / atomsPerBlock);
    dim3 dimBlock(order, order, atomsPerBlock);

    const bool wrapX = true;
    const bool wrapY = true;
    GMX_UNUSED_VALUE(wrapX);
    GMX_UNUSED_VALUE(wrapY);

    pme_gpu_start_timing(pmeGpu, gtPME_GATHER);
    if (order == 4)
    {
        if (overwriteForces)
        {
            pme_gather_kernel<4, true, wrapX, wrapY> <<< nBlocks, dimBlock, 0, stream>>> (*kernelParamsPtr);
        }
        else
        {
            pme_gather_kernel<4, false, wrapX, wrapY> <<< nBlocks, dimBlock, 0, stream>>> (*kernelParamsPtr);
        }
    }
    else
    {
        GMX_THROW(gmx::NotImplementedError("The code for pme_order != 4 is not implemented"));
    }
    CU_LAUNCH_ERR("pme_gather_kernel");
    pme_gpu_stop_timing(pmeGpu, gtPME_GATHER);

    pme_gpu_copy_output_forces(pmeGpu, h_forces);
}
