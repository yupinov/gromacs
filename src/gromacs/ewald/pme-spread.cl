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

/*
 * This define affects the spline calculation behaviour in the kernel.
 * 0: a single GPU thread handles a single dimension of a single particle (calculating and storing (order) spline values and derivatives).
 * 1: (order) threads do redundant work on this same task, each one stores only a single theta and single dtheta into global arrays.
 * The only efficiency difference is less global store operations, countered by more redundant spline computation.
 *
 * TODO: estimate if this should be a boolean parameter (and add it to the unit test if so).
 */
#define PME_GPU_PARALLEL_SPLINE 0

#ifndef COMPILE_SPREAD_HELPERS_ONCE
#define COMPILE_SPREAD_HELPERS_ONCE

/*! \brief
 * General purpose function for loading atom-related data from global to shared memory.
 *
 * \tparam[in] T                 Data type (float/int/...)
 * \tparam[in] atomsPerBlock     Number of atoms processed by a block - should be accounted for in the size of the shared memory array.
 * \tparam[in] dataCountPerAtom  Number of data elements per single atom (e.g. DIM for an rvec coordinates array).
 * \param[in]  kernelParams      Input PME CUDA data in constant memory.
 * \param[out] sm_destination    Shared memory array for output.
 * \param[in]  gm_source         Global memory array for input.
 */

//TODO: stringify?

DEVICE_INLINE
void pme_gpu_stage_atom_data(const struct PmeGpuCudaKernelParams       kernelParams,
                             SHARED float * __restrict__               sm_destination,
                             GLOBAL const float * __restrict__         gm_source,
                             const int                                 dataCountPerAtom) //FIXME template parameter - maybe inline is just fine?!
{
    static_assert(c_usePadding, "With padding disabled, index checking should be fixed to account for spline theta/dtheta per-warp alignment");
    const size_t threadLocalIndex = getThreadLocalIndex3d();
    const size_t localIndex       = threadLocalIndex;
    const size_t globalIndexBase  = get_group_id(0) * atomsPerBlock * dataCountPerAtom;
    //FIXME blockIdx.x * atomsPerBlock * dataCountPerAtom;
    const size_t globalIndex      = globalIndexBase + localIndex;
    const int    globalCheck      = pme_gpu_check_atom_data_index(globalIndex, kernelParams.atoms.nAtoms * dataCountPerAtom);
    if ((localIndex < atomsPerBlock * dataCountPerAtom) & globalCheck)
    {
        assert(isfinite(float(gm_source[globalIndex])));
        sm_destination[localIndex] = gm_source[globalIndex];
    }
}

/*! \brief
 * PME GPU spline parameter and gridline indices calculation.
 * This corresponds to the CPU functions calc_interpolation_idx() and make_bsplines().
 * First stage of the whole kernel.
 *
 * \tparam[in] order                PME interpolation order.
 * \tparam[in] atomsPerBlock        Number of atoms processed by a block - should be accounted for
 *                                  in the sizes of the shared memory arrays.
 * \param[in]  kernelParams         Input PME CUDA data in constant memory.
 * \param[in]  atomIndexOffset      Starting atom index for the execution block w.r.t. global memory.
 * \param[in]  sm_coordinates       Atom coordinates in the shared memory.
 * \param[in]  sm_coefficients      Atom charges/coefficients in the shared memory.
 * \param[out] sm_theta             Atom spline values in the shared memory.
 * \param[out] sm_gridlineIndices   Atom gridline indices in the shared memory.
 */
DEVICE_INLINE void calculate_splines(const struct PmeGpuCudaKernelParams           kernelParams,
                                     const int                                     atomIndexOffset,
                                     SHARED const float * __restrict__             sm_coordinates,
                                     SHARED const float * __restrict__             sm_coefficients,
                                     SHARED float * __restrict__                   sm_theta,
                                     SHARED int * __restrict__                     sm_gridlineIndices,
                                     SHARED float * __restrict__   sm_fractCoords                             //FIXME moved for Intel
#if !                                                                              CAN_USE_BUFFERS_IN_STRUCTS //FIXME doc
                                     ,
                                     GLOBAL float * __restrict__                   gm_theta,
                                     GLOBAL float * __restrict__                   gm_dtheta,
                                     GLOBAL int * __restrict__                     gm_gridlineIndices,
                                     GLOBAL const float * __restrict__             gm_fractShiftsTable,
                                     GLOBAL const int * __restrict__               gm_gridlineIndicesTable
#endif
                                     )
{
#if CAN_USE_BUFFERS_IN_STRUCTS
    /* Global memory pointers for output */
    GLOBAL float * __restrict__       gm_theta                = kernelParams.atoms.d_theta;
    GLOBAL float * __restrict__       gm_dtheta               = kernelParams.atoms.d_dtheta;
    GLOBAL int * __restrict__         gm_gridlineIndices      = kernelParams.atoms.d_gridlineIndices;
    GLOBAL const float * __restrict__ gm_fractShiftsTable     = kernelParams.grid.d_fractShiftsTable;
    GLOBAL const int * __restrict__   gm_gridlineIndicesTable = kernelParams.grid.d_gridlineIndicesTable;
#endif
    /* Fractional coordinates */
    //FIXME moved outside for Intel SHARED float sm_fractCoords[atomsPerBlock * DIM];

    /* Thread index w.r.t. block */
    const int threadLocalIndex = getThreadLocalIndex3d();
    /* Warp index w.r.t. block - could probably be obtained easier? */
    const int warpIndex = threadLocalIndex / warp_size;
    /* Thread index w.r.t. warp */
    const int threadWarpIndex = threadLocalIndex % warp_size;
    /* Atom index w.r.t. warp - alternating 0 1 0 1 .. */
    const int atomWarpIndex = threadWarpIndex % PME_SPREADGATHER_ATOMS_PER_WARP;
    /* Atom index w.r.t. block/shared memory */
    const int atomIndexLocal = warpIndex * PME_SPREADGATHER_ATOMS_PER_WARP + atomWarpIndex;

    /* Atom index w.r.t. global memory */
    const int atomIndexGlobal = atomIndexOffset + atomIndexLocal;
    /* Spline contribution index in one dimension */
    const int orderIndex = threadWarpIndex / (PME_SPREADGATHER_ATOMS_PER_WARP * DIM);
    /* Dimension index */
    const int dimIndex = (threadWarpIndex - orderIndex * (PME_SPREADGATHER_ATOMS_PER_WARP * DIM)) / PME_SPREADGATHER_ATOMS_PER_WARP;

    /* Multi-purpose index of rvec/ivec atom data */
    const int sharedMemoryIndex = atomIndexLocal * DIM + dimIndex;

    /* Spline parameters need a working buffer.
     * With PME_GPU_PARALLEL_SPLINE == 0 it is just a local array of (order) values for some of the threads, which is fine;
     * With PME_GPU_PARALLEL_SPLINE == 1 (order) times more threads are involved, so the shared memory is used to avoid overhead.
     * The buffer's size, striding and indexing are adjusted accordingly.
     * The buffer is accessed with SPLINE_DATA_PTR and SPLINE_DATA macros.
     */
#if PME_GPU_PARALLEL_SPLINE
    #define  splineDataStride  (atomsPerBlock * DIM)
    const int        splineDataIndex   = sharedMemoryIndex;
    SHARED float     sm_splineData[splineDataStride * order];
    SHARED float    *splineDataPtr = sm_splineData;
#else
    #define splineDataStride 1
    const int        splineDataIndex  = 0;
    float            splineData[splineDataStride * order];
    float           *splineDataPtr = splineData;
#endif

#define SPLINE_DATA_PTR(i) (splineDataPtr + ((i) * splineDataStride + splineDataIndex))
#define SPLINE_DATA(i) (*SPLINE_DATA_PTR(i))

    const int localCheck  = (dimIndex < DIM) && (orderIndex < (PME_GPU_PARALLEL_SPLINE ? order : 1));
    const int globalCheck = pme_gpu_check_atom_data_index(atomIndexGlobal, kernelParams.atoms.nAtoms);

    if (localCheck && globalCheck)
    {
        /* Indices interpolation */
        if (orderIndex == 0)
        {
            int           tableIndex, tInt;
            float         n, t;
            const float3  x = vload3(atomIndexLocal, sm_coordinates);

            /* Accessing fields in fshOffset/nXYZ/recipbox/... with dimIndex offset
             * puts them into local memory(!) instead of accessing the constant memory directly.
             * That's the reason for the switch, to unroll explicitly.
             * The commented parts correspond to the 0 components of the recipbox.
             */
            switch (dimIndex)
            {
                case XX:
                    tableIndex  = kernelParams.grid.tablesOffsets[XX];
                    n           = kernelParams.grid.realGridSizeFP[XX];
                    t           = x.x * kernelParams.current.recipBox[dimIndex][XX] + x.y * kernelParams.current.recipBox[dimIndex][YY] + x.z * kernelParams.current.recipBox[dimIndex][ZZ];
                    break;

                case YY:
                    tableIndex  = kernelParams.grid.tablesOffsets[YY];
                    n           = kernelParams.grid.realGridSizeFP[YY];
                    t           = /*x.x * kernelParams.current.recipbox[dimIndex][XX] + */ x.y * kernelParams.current.recipBox[dimIndex][YY] + x.z * kernelParams.current.recipBox[dimIndex][ZZ];
                    break;

                case ZZ:
                    tableIndex  = kernelParams.grid.tablesOffsets[ZZ];
                    n           = kernelParams.grid.realGridSizeFP[ZZ];
                    t           = /*x.x * kernelParams.current.recipbox[dimIndex][XX] + x.y * kernelParams.current.recipbox[dimIndex][YY] + */ x.z * kernelParams.current.recipBox[dimIndex][ZZ];
                    break;
            }
            const float shift = c_pmeMaxUnitcellShift;

            /* Fractional coordinates along box vectors, adding a positive shift to ensure t is positive for triclinic boxes */
            t    = (t + shift) * n;
            tInt = (int)t;
            sm_fractCoords[sharedMemoryIndex] = t - tInt;
            tableIndex                       += tInt;
            assert(tInt >= 0);
            assert(tInt < c_pmeNeighborUnitcellCount * n);
            // TODO have shared table for both parameters to share the fetch, as index is always same?
            // TODO compare texture/LDG performance
            sm_fractCoords[sharedMemoryIndex]    += gm_fractShiftsTable[tableIndex];
            sm_gridlineIndices[sharedMemoryIndex] = gm_gridlineIndicesTable[tableIndex];
            gm_gridlineIndices[atomIndexOffset * DIM + sharedMemoryIndex] = sm_gridlineIndices[sharedMemoryIndex];
        }

        /* B-spline calculation */

        const int chargeCheck = pme_gpu_check_atom_charge(sm_coefficients[atomIndexLocal]);
        if (chargeCheck)
        {
            float       div;
            int         o = orderIndex; // This is an index that is set once for PME_GPU_PARALLEL_SPLINE == 1

            const float dr = sm_fractCoords[sharedMemoryIndex];
            assert(isfinite(dr));

            /* dr is relative offset from lower cell limit */
            *SPLINE_DATA_PTR(order - 1) = 0.0f;
            *SPLINE_DATA_PTR(1)         = dr;
            *SPLINE_DATA_PTR(0)         = 1.0f - dr;

#pragma unroll
            for (int k = 3; k < order; k++)
            {
                div                     = 1.0f / (k - 1.0f);
                *SPLINE_DATA_PTR(k - 1) = div * dr * SPLINE_DATA(k - 2);
#pragma unroll
                for (int l = 1; l < (k - 1); l++)
                {
                    *SPLINE_DATA_PTR(k - l - 1) = div * ((dr + l) * SPLINE_DATA(k - l - 2) + (k - l - dr) * SPLINE_DATA(k - l - 1));
                }
                *SPLINE_DATA_PTR(0) = div * (1.0f - dr) * SPLINE_DATA(0);
            }

            const int thetaGlobalOffsetBase = atomIndexOffset * DIM * order;

            /* Differentiation and storing the spline derivatives (dtheta) */
#if !PME_GPU_PARALLEL_SPLINE
            // With PME_GPU_PARALLEL_SPLINE == 1, o is already set to orderIndex;
            // With PME_GPU_PARALLEL_SPLINE == 0, we loop o over range(order).
#pragma unroll
            for (o = 0; o < order; o++)
#endif
            {
                const int   thetaIndex       = ((o + order * warpIndex) * DIM + dimIndex) * PME_SPREADGATHER_ATOMS_PER_WARP + atomWarpIndex;
                const int   thetaGlobalIndex = thetaGlobalOffsetBase + thetaIndex;

                const float dtheta = ((o > 0) ? SPLINE_DATA(o - 1) : 0.0f) - SPLINE_DATA(o);
                assert(isfinite(dtheta));
                gm_dtheta[thetaGlobalIndex] = dtheta;
            }

            div  = 1.0f / (order - 1.0f);
            *SPLINE_DATA_PTR(order - 1) = div * dr * SPLINE_DATA(order - 2);
#pragma unroll
            for (int k = 1; k < (order - 1); k++)
            {
                *SPLINE_DATA_PTR(order - k - 1) = div * ((dr + k) * SPLINE_DATA(order - k - 2) + (order - k - dr) * SPLINE_DATA(order - k - 1));
            }
            *SPLINE_DATA_PTR(0) = div * (1.0f - dr) * SPLINE_DATA(0);

            /* Storing the spline values (theta) */
#if !PME_GPU_PARALLEL_SPLINE
            // See comment for the loop above
#pragma unroll
            for (o = 0; o < order; o++)
#endif
            {
                const int thetaIndex       = ((o + order * warpIndex) * DIM + dimIndex) * PME_SPREADGATHER_ATOMS_PER_WARP + atomWarpIndex;
                const int thetaGlobalIndex = thetaGlobalOffsetBase + thetaIndex;

                sm_theta[thetaIndex]       = SPLINE_DATA(o);
                assert(isfinite(sm_theta[thetaIndex]));
                gm_theta[thetaGlobalIndex] = SPLINE_DATA(o);
            }
        }
    }
}

/*! \brief
 * Charge spreading onto the grid.
 * This corresponds to the CPU function spread_coefficients_bsplines_thread().
 * Second stage of the whole kernel.
 *
 * \tparam[in] order                PME interpolation order.
 * \tparam[in] wrapX                A boolean which tells if the grid overlap in dimension X should be wrapped.
 * \tparam[in] wrapY                A boolean which tells if the grid overlap in dimension Y should be wrapped.
 * \param[in]  kernelParams         Input PME CUDA data in constant memory.
 * \param[in]  atomIndexOffset      Starting atom index for the execution block w.r.t. global memory.
 * \param[in]  sm_coefficients      Atom coefficents/charges in the shared memory.
 * \param[in]  sm_gridlineIndices   Atom gridline indices in the shared memory.
 * \param[in]  sm_theta             Atom spline values in the shared memory.
 */
DEVICE_INLINE void spread_charges(const struct PmeGpuCudaKernelParams           kernelParams,
                                  int                                           atomIndexOffset,
                                  SHARED const float * __restrict__             sm_coefficients,
                                  SHARED const int * __restrict__               sm_gridlineIndices,
                                  SHARED const float * __restrict__             sm_theta
#if !                                                                           CAN_USE_BUFFERS_IN_STRUCTS
                                  ,
                                  GLOBAL float * __restrict__                   gm_grid
#endif
                                  )
{
#if CAN_USE_BUFFERS_IN_STRUCTS
    /* Global memory pointer to the output grid */
    GLOBAL float * __restrict__ gm_grid = kernelParams.grid.d_realGrid;
#endif

    const int nx  = kernelParams.grid.realGridSize[XX];
    const int ny  = kernelParams.grid.realGridSize[YY];
    const int nz  = kernelParams.grid.realGridSize[ZZ];
    const int pny = kernelParams.grid.realGridSizePadded[YY];
    const int pnz = kernelParams.grid.realGridSizePadded[ZZ];

    const int offx = 0, offy = 0, offz = 0; // unused for now

    const int atomIndexLocal  = getThreadLocalIndex(ZZ);
    const int atomIndexGlobal = atomIndexOffset + atomIndexLocal;

    const int globalCheck = pme_gpu_check_atom_data_index(atomIndexGlobal, kernelParams.atoms.nAtoms);
    const int chargeCheck = pme_gpu_check_atom_charge(sm_coefficients[atomIndexLocal]);
    if (chargeCheck & globalCheck)
    {
        // Spline Y/Z coordinates
        const int ithy   = getThreadLocalIndex(YY);
        const int ithz   = getThreadLocalIndex(XX);
        const int ixBase = sm_gridlineIndices[atomIndexLocal * DIM + XX] - offx;
        int       iy     = sm_gridlineIndices[atomIndexLocal * DIM + YY] - offy + ithy;
        if (wrapY & (iy >= ny))
        {
            iy -= ny;
        }
        int iz  = sm_gridlineIndices[atomIndexLocal * DIM + ZZ] - offz + ithz;
        if (iz >= nz)
        {
            iz -= nz;
        }

        /* Atom index w.r.t. warp - alternating 0 1 0 1 .. */
        const int           atomWarpIndex     = atomIndexLocal % PME_SPREADGATHER_ATOMS_PER_WARP;
        /* Warp index w.r.t. block - could probably be obtained easier? */
        const int           warpIndex         = atomIndexLocal / PME_SPREADGATHER_ATOMS_PER_WARP;
        const int           dimStride         = PME_SPREADGATHER_ATOMS_PER_WARP;
        const int           orderStride       = dimStride * DIM;
        const int           thetaOffsetBase   = orderStride * order * warpIndex + atomWarpIndex;

        const float         thetaZ         = sm_theta[thetaOffsetBase + ithz * orderStride + ZZ * dimStride];
        const float         thetaY         = sm_theta[thetaOffsetBase + ithy * orderStride + YY * dimStride];
        const float         constVal       = thetaZ * thetaY * sm_coefficients[atomIndexLocal];
        assert(isfinite(constVal));
        const int           constOffset       = iy * pnz + iz;
        SHARED const float *sm_thetaX         = sm_theta + (thetaOffsetBase + XX * dimStride);

#pragma unroll
        for (int ithx = 0; (ithx < order); ithx++)
        {
            int ix = ixBase + ithx;
            if (wrapX & (ix >= nx))
            {
                ix -= nx;
            }
            const int gridIndexGlobal = ix * pny * pnz + constOffset;
            assert(isfinite(sm_thetaX[ithx * orderStride]));
            assert(isfinite(gm_grid[gridIndexGlobal]));
            atomicAdd(gm_grid + gridIndexGlobal, sm_thetaX[ithx * orderStride] * constVal);
        }
    }
}

#endif //COMPILE_SPREAD_HELPERS_ONCE

/*! \brief
 * A spline computation and charge spreading kernel function.
 *
 * \tparam[in] order                PME interpolation order.
 * \tparam[in] computeSplines       A boolean which tells if the spline parameter and
 *                                  gridline indices' computation should be performed.
 * \tparam[in] spreadCharges        A boolean which tells if the charge spreading should be performed.
 * \tparam[in] wrapX                A boolean which tells if the grid overlap in dimension X should be wrapped.
 * \tparam[in] wrapY                A boolean which tells if the grid overlap in dimension Y should be wrapped.
 * \param[in]  kernelParams         Input PME CUDA data in constant memory.
 */
#if defined(GMX_PTX_ARCH) //FIXME icnldue config.h and verify
//FIXME- also __attribute__((reqd_work_group_size(CL_SIZE, CL_SIZE, 1)))???
__launch_bounds__(c_spreadMaxThreadsPerBlock)
#endif
__kernel void CUSTOMIZED_KERNEL_NAME(pme_spline_and_spread_kernel)(const struct PmeGpuCudaKernelParams kernelParams
#if !                                                                                                CAN_USE_BUFFERS_IN_STRUCTS
                                                                   ,
                                                                   GLOBAL float * __restrict__       gm_theta,
                                                                   GLOBAL float * __restrict__       gm_dtheta,
                                                                   GLOBAL int * __restrict__         gm_gridlineIndices,
                                                                   GLOBAL float *__restrict__        gm_grid,
                                                                   GLOBAL const float * __restrict__ gm_fractShiftsTable,
                                                                   GLOBAL const int * __restrict__   gm_gridlineIndicesTable,
                                                                   GLOBAL const float * __restrict__ gm_coefficients,
                                                                   GLOBAL const float * __restrict__ gm_coordinates
#endif
                                                                   )
{
#if CAN_USE_BUFFERS_IN_STRUCTS
    GLOBAL float * __restrict__       gm_theta           = kernelParams.atoms.d_theta;
    GLOBAL int * __restrict__         gm_gridlineIndices = kernelParams.atoms.d_gridlineIndices;
    GLOBAL const float * __restrict__ gm_coefficients    = kernelParams.atoms.d_coefficients;
    GLOBAL const float * __restrict__ gm_coordinates     = kernelParams.atoms.d_coordinates;
#endif

    // Gridline indices, ivec
    SHARED int       sm_gridlineIndices[atomsPerBlock * DIM];
    // Charges
    SHARED float     sm_coefficients[atomsPerBlock];
    // Spline values
    SHARED float     sm_theta[atomsPerBlock * DIM * order];
    // Fractional coordinates - only for spline computation
    SHARED float     sm_fractCoords[atomsPerBlock * DIM];
    // Staging coordinates - only for spline computation
    SHARED float     sm_coordinates[DIM * atomsPerBlock];

    const int        atomIndexOffset = getBlockIndex(XX) * atomsPerBlock;

    /* Staging coefficients/charges for both spline and spread */
    pme_gpu_stage_atom_data(kernelParams, sm_coefficients, gm_coefficients, 1);

    if (computeSplines)
    {
        /* Staging coordinates */
        pme_gpu_stage_atom_data(kernelParams, sm_coordinates, gm_coordinates, DIM);

        barrier(CLK_LOCAL_MEM_FENCE);                                    //TODO LOCAL here because we stage into shared mem?
        //__syncthreads(); //FIXME wrap this?
        calculate_splines(kernelParams, atomIndexOffset, sm_coordinates, //FIXME CUDA had a type cast (SHARED const float3 *)sm_coordinates,
                          sm_coefficients, sm_theta, sm_gridlineIndices,
                          sm_fractCoords
#if !CAN_USE_BUFFERS_IN_STRUCTS
                          , gm_theta, gm_dtheta, gm_gridlineIndices, gm_fractShiftsTable, gm_gridlineIndicesTable
#endif
                          );
        gmx_syncwarp();

        //FIXME this is only here for execution of e.g. 32-sized warps on 16-wide hardware
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    else
    {
        /* Staging the data for spread
         * (the data is assumed to be in GPU global memory with proper layout already,
         * as in after running the spline kernel)
         */
        /* Spline data - only thetas (dthetas will only be needed in gather) */
        pme_gpu_stage_atom_data(kernelParams, sm_theta, gm_theta, DIM * order);
        /* Gridline indices */
        //FIXME hack, assumign sizes
        pme_gpu_stage_atom_data(kernelParams, (SHARED float *)sm_gridlineIndices, (GLOBAL const float *)gm_gridlineIndices, DIM);

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    /* Spreading */
    if (spreadCharges)
    {
        spread_charges(kernelParams, atomIndexOffset, sm_coefficients, sm_gridlineIndices, sm_theta
#if !CAN_USE_BUFFERS_IN_STRUCTS
                       , gm_grid
#endif
                       );
    }
}
