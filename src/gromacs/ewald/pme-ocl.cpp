/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017,2018, by the GROMACS development team, led by
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
 * \brief This file contains internal CUDA function implementations
 * for performing the PME calculations on GPU.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include <cmath>

/* The rest */
#include "pme.h"

//#include "gromacs/gpu_utils/cudautils.cuh"
//#include "gromacs/gpu_utils/devicebuffer.cuh"
#include "gromacs/gpu_utils/oclutils.h"
#include "gromacs/gpu_utils/ocl_compiler.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/smalloc.h"

#include "pme-types-ocl.h"
//#include "pme-3dfft.cuh"
#include "pme-grid.h"

int pme_gpu_get_atom_data_alignment(const PmeGpu *pmeGpu)
{
    const int order = pmeGpu->common->pme_order;
    GMX_ASSERT(order > 0, "Invalid PME order");
    return PME_ATOM_DATA_ALIGNMENT;
}

int pme_gpu_get_atoms_per_warp(const PmeGpu *pmeGpu)
{
    const int order = pmeGpu->common->pme_order;
    GMX_ASSERT(order > 0, "Invalid PME order");
    return PME_SPREADGATHER_ATOMS_PER_WARP;
}

void pme_gpu_synchronize(const PmeGpu *pmeGpu)
{
    gpuStreamSynchronize(pmeGpu->archSpecific->pmeStream);
}

void pme_gpu_alloc_energy_virial(const PmeGpu *pmeGpu)
{
    const size_t energyAndVirialSize = c_virialAndEnergyCount * sizeof(float);
    allocateDeviceBuffer(&pmeGpu->kernelParams->constants.d_virialAndEnergy, c_virialAndEnergyCount, pmeGpu->archSpecific->context);
    ocl_pmalloc((void **)&pmeGpu->staging.h_virialAndEnergy, energyAndVirialSize);
}

void pme_gpu_free_energy_virial(PmeGpu *pmeGpu)
{
    freeDeviceBuffer(&pmeGpu->kernelParams->constants.d_virialAndEnergy);
    ocl_pfree(pmeGpu->staging.h_virialAndEnergy);
    pmeGpu->staging.h_virialAndEnergy = nullptr;
}

void pme_gpu_clear_energy_virial(const PmeGpu *pmeGpu)
{
    clearDeviceBufferAsync(&pmeGpu->kernelParams->constants.d_virialAndEnergy, 0,
                           c_virialAndEnergyCount, pmeGpu->archSpecific->pmeStream);
}

void pme_gpu_realloc_and_copy_bspline_values(const PmeGpu *pmeGpu)
{
    const int splineValuesOffset[DIM] = {
        0,
        pmeGpu->kernelParams->grid.realGridSize[XX],
        pmeGpu->kernelParams->grid.realGridSize[XX] + pmeGpu->kernelParams->grid.realGridSize[YY]
    };
    memcpy((void *)&pmeGpu->kernelParams->grid.splineValuesOffset, &splineValuesOffset, sizeof(splineValuesOffset));

    const int newSplineValuesSize = pmeGpu->kernelParams->grid.realGridSize[XX] +
        pmeGpu->kernelParams->grid.realGridSize[YY] +
        pmeGpu->kernelParams->grid.realGridSize[ZZ];
    const bool shouldRealloc = (newSplineValuesSize > pmeGpu->archSpecific->splineValuesSize);
    reallocateDeviceBuffer(&pmeGpu->kernelParams->grid.d_splineModuli, newSplineValuesSize,
                           &pmeGpu->archSpecific->splineValuesSize, &pmeGpu->archSpecific->splineValuesSizeAlloc, pmeGpu->archSpecific->context);
    if (shouldRealloc)
    {
        /* Reallocate the host buffer */
        ocl_pfree(pmeGpu->staging.h_splineModuli);
        ocl_pmalloc((void **)&pmeGpu->staging.h_splineModuli, newSplineValuesSize * sizeof(float));
    }
    for (int i = 0; i < DIM; i++)
    {
        memcpy(pmeGpu->staging.h_splineModuli + splineValuesOffset[i], pmeGpu->common->bsp_mod[i].data(), pmeGpu->common->bsp_mod[i].size() * sizeof(float));
    }
    /* TODO: pin original buffer instead! */
    copyToDeviceBuffer(&pmeGpu->kernelParams->grid.d_splineModuli, pmeGpu->staging.h_splineModuli, 0,
                newSplineValuesSize, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
}

void pme_gpu_free_bspline_values(const PmeGpu *pmeGpu)
{
    ocl_pfree(pmeGpu->staging.h_splineModuli);
    freeDeviceBuffer(&pmeGpu->kernelParams->grid.d_splineModuli);
}

void pme_gpu_realloc_forces(PmeGpu *pmeGpu)
{
    const size_t newForcesSize = pmeGpu->nAtomsAlloc * DIM;
    GMX_ASSERT(newForcesSize > 0, "Bad number of atoms in PME GPU");
    reallocateDeviceBuffer(&pmeGpu->kernelParams->atoms.d_forces, newForcesSize,
                           &pmeGpu->archSpecific->forcesSize, &pmeGpu->archSpecific->forcesSizeAlloc, pmeGpu->archSpecific->context);
    pmeGpu->staging.h_forces.reserve(pmeGpu->nAtomsAlloc);
    pmeGpu->staging.h_forces.resize(pmeGpu->kernelParams->atoms.nAtoms);
}

void pme_gpu_free_forces(const PmeGpu *pmeGpu)
{
    freeDeviceBuffer(&pmeGpu->kernelParams->atoms.d_forces);
}

template <typename Thing>
inline float *FIXMEcast(Thing *thing)
{
    return (float *)thing;
}

void pme_gpu_copy_input_forces(PmeGpu *pmeGpu)
{
    const size_t forcesSize = DIM * pmeGpu->kernelParams->atoms.nAtoms;
    GMX_ASSERT(forcesSize > 0, "Bad number of atoms in PME GPU");
    copyToDeviceBuffer(&pmeGpu->kernelParams->atoms.d_forces, FIXMEcast(pmeGpu->staging.h_forces.data()), 0,
                       forcesSize, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
}

void pme_gpu_copy_output_forces(PmeGpu *pmeGpu)
{
    const size_t forcesSize   = DIM * pmeGpu->kernelParams->atoms.nAtoms;
    GMX_ASSERT(forcesSize > 0, "Bad number of atoms in PME GPU");
    copyFromDeviceBuffer(FIXMEcast(pmeGpu->staging.h_forces.data()), &pmeGpu->kernelParams->atoms.d_forces, 0,
                         forcesSize, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
}

void pme_gpu_realloc_coordinates(const PmeGpu *pmeGpu)
{
    const size_t newCoordinatesSize = pmeGpu->nAtomsAlloc * DIM;
    GMX_ASSERT(newCoordinatesSize > 0, "Bad number of atoms in PME GPU");
    reallocateDeviceBuffer(&pmeGpu->kernelParams->atoms.d_coordinates, newCoordinatesSize,
                           &pmeGpu->archSpecific->coordinatesSize, &pmeGpu->archSpecific->coordinatesSizeAlloc, pmeGpu->archSpecific->context);
    if (c_usePadding)
    {
        const size_t paddingIndex = DIM * pmeGpu->kernelParams->atoms.nAtoms;
        const size_t paddingCount = DIM * pmeGpu->nAtomsAlloc - paddingIndex;
        if (paddingCount > 0)
        {
            clearDeviceBufferAsync(&pmeGpu->kernelParams->atoms.d_coordinates, paddingIndex,
                                   paddingCount, pmeGpu->archSpecific->pmeStream);
        }
    }
}

void pme_gpu_copy_input_coordinates(const PmeGpu *pmeGpu, const rvec *h_coordinates)
{
    GMX_ASSERT(h_coordinates, "Bad host-side coordinate buffer in PME GPU");
#if GMX_DOUBLE
    GMX_RELEASE_ASSERT(false, "Only single precision is supported");
    GMX_UNUSED_VALUE(h_coordinates);
#else
    copyToDeviceBuffer(&pmeGpu->kernelParams->atoms.d_coordinates, FIXMEcast(h_coordinates), 0,
                pmeGpu->kernelParams->atoms.nAtoms * DIM, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
#endif
}

void pme_gpu_free_coordinates(const PmeGpu *pmeGpu)
{
    freeDeviceBuffer(&pmeGpu->kernelParams->atoms.d_coordinates);
}

void pme_gpu_realloc_and_copy_input_coefficients(const PmeGpu *pmeGpu, const float *h_coefficients)
{
    GMX_ASSERT(h_coefficients, "Bad host-side charge buffer in PME GPU");
    const size_t newCoefficientsSize = pmeGpu->nAtomsAlloc;
    GMX_ASSERT(newCoefficientsSize > 0, "Bad number of atoms in PME GPU");
    reallocateDeviceBuffer(&pmeGpu->kernelParams->atoms.d_coefficients, newCoefficientsSize,
                           &pmeGpu->archSpecific->coefficientsSize, &pmeGpu->archSpecific->coefficientsSizeAlloc, pmeGpu->archSpecific->context);
    copyToDeviceBuffer(&pmeGpu->kernelParams->atoms.d_coefficients, FIXMEcast(h_coefficients), 0,
                pmeGpu->kernelParams->atoms.nAtoms, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
    if (c_usePadding)
    {
        const size_t paddingIndex = pmeGpu->kernelParams->atoms.nAtoms;
        const size_t paddingCount = pmeGpu->nAtomsAlloc - paddingIndex;
        if (paddingCount > 0)
        {
            clearDeviceBufferAsync(&pmeGpu->kernelParams->atoms.d_coefficients, paddingIndex,
                                   paddingCount, pmeGpu->archSpecific->pmeStream);
        }
    }
}

void pme_gpu_free_coefficients(const PmeGpu *pmeGpu)
{
    freeDeviceBuffer(&pmeGpu->kernelParams->atoms.d_coefficients);
}

void pme_gpu_realloc_spline_data(const PmeGpu *pmeGpu)
{
    const int    order             = pmeGpu->common->pme_order;
    const int    alignment         = pme_gpu_get_atoms_per_warp(pmeGpu);
    const size_t nAtomsPadded      = ((pmeGpu->nAtomsAlloc + alignment - 1) / alignment) * alignment;
    const int    newSplineDataSize = DIM * order * nAtomsPadded;
    GMX_ASSERT(newSplineDataSize > 0, "Bad number of atoms in PME GPU");
    /* Two arrays of the same size */
    const bool shouldRealloc        = (newSplineDataSize > pmeGpu->archSpecific->splineDataSize);
    int        currentSizeTemp      = pmeGpu->archSpecific->splineDataSize;
    int        currentSizeTempAlloc = pmeGpu->archSpecific->splineDataSizeAlloc;
    reallocateDeviceBuffer(&pmeGpu->kernelParams->atoms.d_theta, newSplineDataSize,
                           &currentSizeTemp, &currentSizeTempAlloc, pmeGpu->archSpecific->context);
    reallocateDeviceBuffer(&pmeGpu->kernelParams->atoms.d_dtheta, newSplineDataSize,
                           &pmeGpu->archSpecific->splineDataSize, &pmeGpu->archSpecific->splineDataSizeAlloc, pmeGpu->archSpecific->context);
    // the host side reallocation
    if (shouldRealloc)
    {
        ocl_pfree(pmeGpu->staging.h_theta);
        ocl_pmalloc((void **)&pmeGpu->staging.h_theta, newSplineDataSize * sizeof(float));
        ocl_pfree(pmeGpu->staging.h_dtheta);
        ocl_pmalloc((void **)&pmeGpu->staging.h_dtheta, newSplineDataSize * sizeof(float));
    }
}

void pme_gpu_free_spline_data(const PmeGpu *pmeGpu)
{
    /* Two arrays of the same size */
    freeDeviceBuffer(&pmeGpu->kernelParams->atoms.d_theta);
    freeDeviceBuffer(&pmeGpu->kernelParams->atoms.d_dtheta);
    ocl_pfree(pmeGpu->staging.h_theta);
    ocl_pfree(pmeGpu->staging.h_dtheta);
}

void pme_gpu_realloc_grid_indices(const PmeGpu *pmeGpu)
{
    const size_t newIndicesSize = DIM * pmeGpu->nAtomsAlloc;
    GMX_ASSERT(newIndicesSize > 0, "Bad number of atoms in PME GPU");
    reallocateDeviceBuffer(&pmeGpu->kernelParams->atoms.d_gridlineIndices, newIndicesSize,
                           &pmeGpu->archSpecific->gridlineIndicesSize, &pmeGpu->archSpecific->gridlineIndicesSizeAlloc, pmeGpu->archSpecific->context);
    ocl_pfree(pmeGpu->staging.h_gridlineIndices);
    ocl_pmalloc((void **)&pmeGpu->staging.h_gridlineIndices, newIndicesSize * sizeof(int));
}

void pme_gpu_free_grid_indices(const PmeGpu *pmeGpu)
{
    freeDeviceBuffer(&pmeGpu->kernelParams->atoms.d_gridlineIndices);
    ocl_pfree(pmeGpu->staging.h_gridlineIndices);
}

void pme_gpu_realloc_grids(PmeGpu *pmeGpu)
{
    auto     *kernelParamsPtr = pmeGpu->kernelParams.get();
    const int newRealGridSize = kernelParamsPtr->grid.realGridSizePadded[XX] *
        kernelParamsPtr->grid.realGridSizePadded[YY] *
        kernelParamsPtr->grid.realGridSizePadded[ZZ];
    const int newComplexGridSize = kernelParamsPtr->grid.complexGridSizePadded[XX] *
        kernelParamsPtr->grid.complexGridSizePadded[YY] *
        kernelParamsPtr->grid.complexGridSizePadded[ZZ] * 2;
    // Multiplied by 2 because we count complex grid size for complex numbers, but all allocations/pointers are float
    if (pmeGpu->archSpecific->performOutOfPlaceFFT)
    {
        /* 2 separate grids */
        reallocateDeviceBuffer(&kernelParamsPtr->grid.d_fourierGrid, newComplexGridSize,
                               &pmeGpu->archSpecific->complexGridSize, &pmeGpu->archSpecific->complexGridSizeAlloc, pmeGpu->archSpecific->context);
        reallocateDeviceBuffer(&kernelParamsPtr->grid.d_realGrid, newRealGridSize,
                               &pmeGpu->archSpecific->realGridSize, &pmeGpu->archSpecific->realGridSizeAlloc, pmeGpu->archSpecific->context);
    }
    else
    {
        /* A single buffer so that any grid will fit */
        const int newGridsSize = std::max(newRealGridSize, newComplexGridSize);
        reallocateDeviceBuffer(&kernelParamsPtr->grid.d_realGrid, newGridsSize,
                               &pmeGpu->archSpecific->realGridSize, &pmeGpu->archSpecific->realGridSizeAlloc, pmeGpu->archSpecific->context);
        kernelParamsPtr->grid.d_fourierGrid   = kernelParamsPtr->grid.d_realGrid;
        pmeGpu->archSpecific->complexGridSize = pmeGpu->archSpecific->realGridSize;
        // the size might get used later for copying the grid
    }
}

void pme_gpu_free_grids(const PmeGpu *pmeGpu)
{
    if (pmeGpu->archSpecific->performOutOfPlaceFFT)
    {
        freeDeviceBuffer(&pmeGpu->kernelParams->grid.d_fourierGrid);
    }
    freeDeviceBuffer(&pmeGpu->kernelParams->grid.d_realGrid);
}

void pme_gpu_clear_grids(const PmeGpu *pmeGpu)
{
    clearDeviceBufferAsync(&pmeGpu->kernelParams->grid.d_realGrid, 0,
                           pmeGpu->archSpecific->realGridSize, pmeGpu->archSpecific->pmeStream);
}

void pme_gpu_realloc_and_copy_fract_shifts(PmeGpu *pmeGpu)
{
    pme_gpu_free_fract_shifts(pmeGpu);

    auto        *kernelParamsPtr = pmeGpu->kernelParams.get();

    const int    nx                  = kernelParamsPtr->grid.realGridSize[XX];
    const int    ny                  = kernelParamsPtr->grid.realGridSize[YY];
    const int    nz                  = kernelParamsPtr->grid.realGridSize[ZZ];
    const int    cellCount           = c_pmeNeighborUnitcellCount;
    const int    gridDataOffset[DIM] = {0, cellCount * nx, cellCount * (nx + ny)};

    memcpy(kernelParamsPtr->grid.tablesOffsets, &gridDataOffset, sizeof(gridDataOffset));

    const int    newFractShiftsSize  = cellCount * (nx + ny + nz);

    initParamLookupTable(&kernelParamsPtr->grid.d_fractShiftsTable,
                         kernelParamsPtr->fractShiftsTableTexture,
                         pmeGpu->common->fsh.data(),
                         newFractShiftsSize,
                         pmeGpu->deviceInfo,
                         pmeGpu->archSpecific->context);

    initParamLookupTable(&kernelParamsPtr->grid.d_gridlineIndicesTable,
                         kernelParamsPtr->gridlineIndicesTableTexture,
                         pmeGpu->common->nn.data(),
                         newFractShiftsSize,
                         pmeGpu->deviceInfo,
                         pmeGpu->archSpecific->context);
}

void pme_gpu_free_fract_shifts(const PmeGpu *pmeGpu)
{
    auto *kernelParamsPtr = pmeGpu->kernelParams.get();
    destroyParamLookupTable(&kernelParamsPtr->grid.d_fractShiftsTable,
                            kernelParamsPtr->fractShiftsTableTexture,
                            pmeGpu->deviceInfo);
    destroyParamLookupTable(&kernelParamsPtr->grid.d_gridlineIndicesTable,
                            kernelParamsPtr->gridlineIndicesTableTexture,
                            pmeGpu->deviceInfo);
}

bool pme_gpu_stream_query(const PmeGpu *pmeGpu)
{
    return haveStreamTasksCompleted(pmeGpu->archSpecific->pmeStream);
}

void pme_gpu_copy_input_gather_grid(const PmeGpu *pmeGpu, float *h_grid)
{
    copyToDeviceBuffer(&pmeGpu->kernelParams->grid.d_realGrid, h_grid, 0,
                       pmeGpu->archSpecific->realGridSize, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
}

void pme_gpu_copy_output_spread_grid(const PmeGpu *pmeGpu, float *h_grid)
{
    copyFromDeviceBuffer(h_grid, &pmeGpu->kernelParams->grid.d_realGrid, 0, pmeGpu->archSpecific->realGridSize,
                         pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
    //FIXME just pass it directly into API instead?
    pmeGpu->archSpecific->syncSpreadGridD2H.markSyncEvent(pmeGpu->archSpecific->pmeStream);
}

void pme_gpu_copy_output_spread_atom_data(PmeGpu *pmeGpu)
{
    const int    alignment       = pme_gpu_get_atoms_per_warp(pmeGpu);
    const size_t nAtomsPadded    = ((pmeGpu->nAtomsAlloc + alignment - 1) / alignment) * alignment;
    const size_t splinesSize     = DIM * nAtomsPadded * pmeGpu->common->pme_order;
    auto        *kernelParamsPtr = pmeGpu->kernelParams.get();

    copyFromDeviceBuffer(pmeGpu->staging.h_dtheta, &kernelParamsPtr->atoms.d_dtheta, 0,
                         splinesSize, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
    copyFromDeviceBuffer(pmeGpu->staging.h_theta, &kernelParamsPtr->atoms.d_theta, 0,
                         splinesSize, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
    copyFromDeviceBuffer(pmeGpu->staging.h_gridlineIndices, &kernelParamsPtr->atoms.d_gridlineIndices, 0,
                         kernelParamsPtr->atoms.nAtoms * DIM, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
}

void pme_gpu_copy_input_gather_atom_data(const PmeGpu *pmeGpu)
{
    const int    alignment       = pme_gpu_get_atoms_per_warp(pmeGpu);
    const size_t nAtomsPadded    = ((pmeGpu->nAtomsAlloc + alignment - 1) / alignment) * alignment;
    const size_t splinesSize     = DIM * nAtomsPadded * pmeGpu->common->pme_order;
    auto        *kernelParamsPtr = pmeGpu->kernelParams.get();
    if (c_usePadding)
    {
        const size_t splineValuesPerAtom = pmeGpu->common->pme_order * DIM;
        // TODO: could clear only the padding and not the whole thing, but this is a test-exclusive code anyway
        clearDeviceBufferAsync(&kernelParamsPtr->atoms.d_gridlineIndices, 0,
                                pmeGpu->nAtomsAlloc * DIM, pmeGpu->archSpecific->pmeStream);
        clearDeviceBufferAsync(&kernelParamsPtr->atoms.d_dtheta, 0,
                                pmeGpu->nAtomsAlloc * splineValuesPerAtom, pmeGpu->archSpecific->pmeStream);
        clearDeviceBufferAsync(&kernelParamsPtr->atoms.d_theta, 0,
                                pmeGpu->nAtomsAlloc * splineValuesPerAtom, pmeGpu->archSpecific->pmeStream);
    }
    copyToDeviceBuffer(&kernelParamsPtr->atoms.d_dtheta, pmeGpu->staging.h_dtheta, 0,
                       splinesSize, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
    copyToDeviceBuffer(&kernelParamsPtr->atoms.d_theta, pmeGpu->staging.h_theta, 0,
                       splinesSize, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
    copyToDeviceBuffer(&kernelParamsPtr->atoms.d_gridlineIndices, pmeGpu->staging.h_gridlineIndices, 0,
                kernelParamsPtr->atoms.nAtoms * DIM, pmeGpu->archSpecific->pmeStream, pmeGpu->settings.transferKind, nullptr);
}

void pme_gpu_sync_spread_grid(const PmeGpu *pmeGpu)
{
    pmeGpu->archSpecific->syncSpreadGridD2H.waitForSyncEvent(pmeGpu->archSpecific->pmeStream);
}

#if GMX_GPU == GMX_GPU_OPENCL
// based on nbnxn_gpu_compile_kernels
void pme_gpu_compile_kernels(PmeGpu *pmeGpu)
{
    cl_program program  = nullptr;
    /* Need to catch std::bad_alloc here and during compilation string
       handling. */
    try
    {
        /* Here we pass macros and static const int variables defined in include
         * files outside the nbnxn_ocl as macros, to avoid including those files
         * in the JIT compilation that happens at runtime.
         */

        const std::string defines = "";
        #if 0
                * gmx::formatString(
                    " -DCENTRAL=%d "
                    "-DNBNXN_GPU_NCLUSTER_PER_SUPERCLUSTER=%d -DNBNXN_GPU_CLUSTER_SIZE=%d -DNBNXN_GPU_JGROUP_SIZE=%d "
                    "-DGMX_NBNXN_PRUNE_KERNEL_J4_CONCURRENCY=%d "
                    "-DNBNXN_MIN_RSQ=%s %s",
                    CENTRAL,                                                /* Defined in ishift.h */
                    c_nbnxnGpuNumClusterPerSupercluster,                    /* Defined in nbnxn_pairlist.h */
                    c_nbnxnGpuClusterSize,                                  /* Defined in nbnxn_pairlist.h */
                    c_nbnxnGpuJgroupSize,                                   /* Defined in nbnxn_pairlist.h */
                    getOclPruneKernelJ4Concurrency(nb->dev_info->vendor_e), /* In nbnxn_ocl_types.h  */
                    STRINGIFY_MACRO(NBNXN_MIN_RSQ)                          /* Defined in nbnxn_consts.h */
                                                                            /* NBNXN_MIN_RSQ passed as string to avoid
                                                                                floating point representation problems with sprintf */
                    , (nb->bPrefetchLjParam) ? "-DIATYPE_SHMEM" : ""
                    );
#endif

        try
        {
            /* TODO when we have a proper MPI-aware logging module,
               the log output here should be written there */
            program = gmx::ocl::compileProgram(stderr,
                                               "pme-spread-kernel.cl",
                                               defines,
                                               pmeGpu->archSpecific->context,
                                               pmeGpu->deviceInfo->ocl_gpu_id.ocl_device_id,
                                               pmeGpu->deviceInfo->vendor_e);
        }
        catch (gmx::GromacsException &e)
        {
            e.prependContext(gmx::formatString("Failed to compile PME kernels for GPU #%s\n",
                                               pmeGpu->deviceInfo->device_name));
            throw;
        }
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;

    pmeGpu->archSpecific->program = program;
}
#endif

void pme_gpu_init_internal(PmeGpu *pmeGpu)
{
    /* Allocate the target-specific structures */
    pmeGpu->archSpecific.reset(new PmeGpuSpecific());
    pmeGpu->kernelParams.reset(new PmeGpuKernelParams());

    pmeGpu->archSpecific->performOutOfPlaceFFT = true;
    /* This should give better performance, according to the cuFFT documentation.
     * The performance seems to be the same though.
     * TODO: PME could also try to pick up nice grid sizes (with factors of 2, 3, 5, 7).
     */

    //FIXME all this code borrowed from nbnxn_
    cl_context_properties     context_properties[3];
    cl_platform_id            platform_id;
    cl_device_id              device_id;
    cl_int                    cl_error;

    platform_id      = pmeGpu->deviceInfo->ocl_gpu_id.ocl_platform_id;
    device_id        = pmeGpu->deviceInfo->ocl_gpu_id.ocl_device_id;

    context_properties[0] = CL_CONTEXT_PLATFORM;
    context_properties[1] = (cl_context_properties) platform_id;
    context_properties[2] = 0; /* Terminates the list of properties */

    pmeGpu->archSpecific->context = clCreateContext(context_properties, 1, &device_id, NULL, NULL, &cl_error);
    GMX_RELEASE_ASSERT(CL_SUCCESS == cl_error, "whatever");
                     /*
                       gmx::formatString("Failed to create context for PME on GPU #%s:\n OpenCL error %d: %s",
                  pmeGpu->deviceInfo->device_name,
                  cl_error, ocl_get_error_string(cl_error).c_str())*/

    /* WARNING: CUDA timings are incorrect with multiple streams.
     *          This is the main reason why they are disabled by default.
     */
    // TODO: Consider turning on by default when we can detect nr of streams.
    pmeGpu->archSpecific->useTiming = (getenv("GMX_ENABLE_GPU_TIMING") != nullptr); //FIXME: DISABLE for OpenCL?

    /* Creating a GPU stream */

    // TODO wrapper;
    // TODO priorities/out of order? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
#if GMX_GPU == GMX_GPU_OPENCL
    cl_command_queue_properties queueProperties = pmeGpu->archSpecific->useTiming ? CL_QUEUE_PROFILING_ENABLE : 0;

    /* local/non-local GPU streams */
    pmeGpu->archSpecific->pmeStream = clCreateCommandQueue(pmeGpu->archSpecific->context,
                                                           device_id, queueProperties, &cl_error);
    GMX_RELEASE_ASSERT(cl_error == CL_SUCCESS, "Failed to create command queue");
#endif
#if GMX_GPU == GMX_GPU_CUDA
    cudaError_t stat;
    int         highest_priority, lowest_priority;
    stat = cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
    CU_RET_ERR(stat, "PME cudaDeviceGetStreamPriorityRange failed");
    stat = cudaStreamCreateWithPriority(&pmeGpu->archSpecific->pmeStream,
                                        cudaStreamDefault, //cudaStreamNonBlocking,
                                        highest_priority);
    CU_RET_ERR(stat, "cudaStreamCreateWithPriority on the PME stream failed");
#endif

#if GMX_GPU == GMX_GPU_OPENCL
    pme_gpu_compile_kernels(pmeGpu);
#endif
}

void pme_gpu_destroy_specific(const PmeGpu *pmeGpu)
{
    /* Free command queues */
    cl_int clError = clReleaseCommandQueue(pmeGpu->archSpecific->pmeStream);
    GMX_RELEASE_ASSERT(clError == CL_SUCCESS, "PME stream destruction error");

    //FIXME
    /*
    free_gpu_device_runtime_data(nb->dev_rundata);
    sfree(nb->dev_rundata);
    */
    clError = clReleaseContext(pmeGpu->archSpecific->context);
    GMX_RELEASE_ASSERT(clError == CL_SUCCESS, "PME context destruction error");
}

#ifdef CLFFTFOUND
void pme_gpu_reinit_3dfft(const PmeGpu *pmeGpu)
{
    if (pme_gpu_performs_FFT(pmeGpu))
    {
        pmeGpu->archSpecific->fftSetup.resize(0);
        for (int i = 0; i < pmeGpu->common->ngrids; i++)
        {
            pmeGpu->archSpecific->fftSetup.push_back(std::unique_ptr<GpuParallel3dFft>(new GpuParallel3dFft(pmeGpu)));
        }
    }
}

void pme_gpu_destroy_3dfft(const PmeGpu *pmeGpu)
{
    pmeGpu->archSpecific->fftSetup.resize(0);
}
#else
struct GpuParallel3dFft
{
    //hello I am dummy!
};
#endif
