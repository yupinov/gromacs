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
 * \brief This file contains internal CUDA function implementations
 * for performing the PME calculations on GPU.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

/* GPU initialization includes */
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/hardware/gpu_hw_info.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/logger.h"

/* The rest */
#include "pme.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/pmalloc_cuda.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/smalloc.h"

#include "pme.cuh"
#include "pme-3dfft.cuh"
#include "pme-grid.h"

int pme_gpu_get_atom_data_alignment(const pme_gpu_t *pmeGPU)
{
    const int order = pmeGPU->common.get()->pme_order;
    GMX_RELEASE_ASSERT(order > 0, "Invalid PME order");
    return PME_SPREADGATHER_ATOMS_PER_BLOCK;
}


#define PME_GPU_PAGELOCKING_HACK 1
/* Enables temporary hack for PME GPU to realloc with page size alignment
 * and pagelock the important host pointers (forces, charges, coordinates...)
 * so that the CUDA copies are non-blocking.
 * TODO: this should be removed when the AlignedAllocator learns how to page-lock.
 */

void pme_gpu_synchronize(const pme_gpu_t *pmeGPU)
{
    cudaError_t stat = cudaStreamSynchronize(pmeGPU->archSpecific->pmeStream);
    CU_RET_ERR(stat, "Failed to synchronize the PME GPU stream!");
}

void pme_gpu_alloc_energy_virial(const pme_gpu_t *pmeGPU)
{
    const size_t energyAndVirialSize = PME_GPU_VIRIAL_AND_ENERGY_COUNT * sizeof(float);
    cudaError_t  stat                = cudaMalloc((void **)&pmeGPU->kernelParams.get()->constants.d_virialAndEnergy, energyAndVirialSize);
    CU_RET_ERR(stat, "cudaMalloc failed on PME energy and virial");
    pmalloc((void **)&pmeGPU->staging.h_virialAndEnergy, energyAndVirialSize);
}

void pme_gpu_free_energy_virial(pme_gpu_t *pmeGPU)
{
    cudaError_t stat = cudaFree(pmeGPU->kernelParams.get()->constants.d_virialAndEnergy);
    CU_RET_ERR(stat, "cudaFree failed on PME energy and virial");
    pmeGPU->kernelParams.get()->constants.d_virialAndEnergy = NULL;
    pfree(pmeGPU->staging.h_virialAndEnergy);
    pmeGPU->staging.h_virialAndEnergy = NULL;
}

void pme_gpu_clear_energy_virial(const pme_gpu_t *pmeGPU)
{
    cudaError_t stat = cudaMemsetAsync(pmeGPU->kernelParams.get()->constants.d_virialAndEnergy, 0,
                                       PME_GPU_VIRIAL_AND_ENERGY_COUNT * sizeof(float), pmeGPU->archSpecific->pmeStream);
    CU_RET_ERR(stat, "PME energies/virial cudaMemsetAsync error");
}

void pme_gpu_realloc_and_copy_bspline_values(const pme_gpu_t *pmeGPU)
{
    const int splineValuesOffset[DIM] = {
        0,
        pmeGPU->kernelParams.get()->grid.localGridSize[XX],
        pmeGPU->kernelParams.get()->grid.localGridSize[XX] + pmeGPU->kernelParams.get()->grid.localGridSize[YY]
    };
    memcpy((void *)&pmeGPU->kernelParams.get()->grid.splineValuesOffset, &splineValuesOffset, sizeof(splineValuesOffset));

    const int newSplineValuesSize = pmeGPU->kernelParams.get()->grid.localGridSize[XX] +
        pmeGPU->kernelParams.get()->grid.localGridSize[YY] +
        pmeGPU->kernelParams.get()->grid.localGridSize[ZZ];
    cu_realloc_buffered((void **)&pmeGPU->kernelParams.get()->grid.d_splineModuli, NULL, sizeof(float),
                        &pmeGPU->archSpecific->splineValuesSize, &pmeGPU->archSpecific->splineValuesSizeAlloc, newSplineValuesSize, pmeGPU->archSpecific->pmeStream, true);

    for (int i = 0; i < DIM; i++)
    {
        /* Reallocate the host buffer */
        const size_t modSize = pmeGPU->kernelParams.get()->grid.localGridSize[i] * sizeof(float);
        if ((pmeGPU->staging.h_splineModuli[i] == NULL) || (pmeGPU->staging.splineModuliSizes[i] < modSize))
        {
            pfree(pmeGPU->staging.h_splineModuli[i]);
            pmalloc((void **)&pmeGPU->staging.h_splineModuli[i], modSize);
        }
        memcpy((void *)pmeGPU->staging.h_splineModuli[i], pmeGPU->common->bsp_mod[i].data(), modSize);
        /* TODO: use pinning here as well! */
        /* FIXME: no need for separate buffers */
        cu_copy_H2D_async(pmeGPU->kernelParams.get()->grid.d_splineModuli + splineValuesOffset[i], pmeGPU->staging.h_splineModuli[i], modSize, pmeGPU->archSpecific->pmeStream);
    }
}

void pme_gpu_free_bspline_values(const pme_gpu_t *pmeGPU)
{
    for (int i = 0; i < DIM; i++)
    {
        pfree(pmeGPU->staging.h_splineModuli[i]);
    }
    cu_free_buffered(pmeGPU->kernelParams.get()->grid.d_splineModuli, &pmeGPU->archSpecific->splineValuesSize, &pmeGPU->archSpecific->splineValuesSizeAlloc);
}

void pme_gpu_realloc_forces(const pme_gpu_t *pmeGPU)
{
    const size_t newForcesSize = pmeGPU->nAtomsAlloc * DIM;
    GMX_ASSERT(newForcesSize > 0, "Bad number of atoms in PME GPU");
    cu_realloc_buffered((void **)&pmeGPU->kernelParams.get()->atoms.d_forces, NULL, sizeof(float),
                        &pmeGPU->archSpecific->forcesSize, &pmeGPU->archSpecific->forcesSizeAlloc, newForcesSize, pmeGPU->archSpecific->pmeStream, true);
}

void pme_gpu_free_forces(const pme_gpu_t *pmeGPU)
{
    cu_free_buffered(pmeGPU->kernelParams.get()->atoms.d_forces, &pmeGPU->archSpecific->forcesSize, &pmeGPU->archSpecific->forcesSizeAlloc);
}

void pme_gpu_copy_input_forces(const pme_gpu_t *pmeGPU, const float *h_forces)
{
    GMX_ASSERT(h_forces, "NULL host forces pointer in PME GPU");
    const size_t forcesSize = DIM * pmeGPU->kernelParams.get()->atoms.nAtoms * sizeof(float);
    GMX_ASSERT(forcesSize > 0, "Bad number of atoms in PME GPU");
    pme_gpu_make_sure_memory_is_pinned((void **)&h_forces, forcesSize);
    cu_copy_H2D_async(pmeGPU->kernelParams.get()->atoms.d_forces, const_cast<float *>(h_forces), forcesSize, pmeGPU->archSpecific->pmeStream);
}

void pme_gpu_copy_output_forces(const pme_gpu_t *pmeGPU, float *h_forces)
{
    GMX_ASSERT(h_forces, "NULL host forces pointer in PME GPU");
    const size_t forcesSize   = DIM * pmeGPU->kernelParams.get()->atoms.nAtoms * sizeof(float);
    GMX_ASSERT(forcesSize > 0, "Bad number of atoms in PME GPU");
    pme_gpu_make_sure_memory_is_pinned((void **)&h_forces, forcesSize);
    cu_copy_D2H_async(h_forces, pmeGPU->kernelParams.get()->atoms.d_forces, forcesSize, pmeGPU->archSpecific->pmeStream);
    cudaError_t stat = cudaEventRecord(pmeGPU->archSpecific->syncForcesD2H, pmeGPU->archSpecific->pmeStream);
    CU_RET_ERR(stat, "PME gather forces synchronization failure");
}

void pme_gpu_sync_output_forces(const pme_gpu_t *pmeGPU)
{
    cudaStream_t s    = pmeGPU->archSpecific->pmeStream;
    cudaError_t  stat = cudaStreamWaitEvent(s, pmeGPU->archSpecific->syncForcesD2H, 0);
    CU_RET_ERR(stat, "Error while waiting for the PME GPU forces");
}

void pme_gpu_realloc_coordinates(const pme_gpu_t *pmeGPU)
{
    const size_t newCoordinatesSize = pmeGPU->nAtomsAlloc * DIM;
    GMX_ASSERT(newCoordinatesSize > 0, "Bad number of atoms in PME GPU");
    cu_realloc_buffered((void **)&pmeGPU->kernelParams.get()->atoms.d_coordinates, NULL, sizeof(float),
                        &pmeGPU->archSpecific->coordinatesSize, &pmeGPU->archSpecific->coordinatesSizeAlloc, newCoordinatesSize, pmeGPU->archSpecific->pmeStream, true);
#if PME_GPU_USE_PADDING
    const size_t paddingIndex = DIM * pmeGPU->kernelParams.get()->atoms.nAtoms;
    const size_t paddingCount = DIM * pmeGPU->nAtomsAlloc - paddingIndex;
    if (paddingCount > 0)
    {
        cudaError_t stat = cudaMemsetAsync(pmeGPU->kernelParams.get()->atoms.d_coordinates + paddingIndex, 0, paddingCount * sizeof(float), pmeGPU->archSpecific->pmeStream);
        CU_RET_ERR(stat, "PME failed to clear the padded coordinates");
    }
#endif
}

void pme_gpu_copy_input_coordinates(const pme_gpu_t *pmeGPU, const rvec *h_coordinates)
{
    GMX_ASSERT(h_coordinates, "Bad host-side coordinate buffer in PME GPU");
    GMX_RELEASE_ASSERT(sizeof(real) == sizeof(float), "Only single precision supported");
    pme_gpu_make_sure_memory_is_pinned((void **)&h_coordinates, pmeGPU->kernelParams.get()->atoms.nAtoms * sizeof(rvec));
    cu_copy_H2D_async(pmeGPU->kernelParams.get()->atoms.d_coordinates, const_cast<rvec *>(h_coordinates),
                      pmeGPU->kernelParams.get()->atoms.nAtoms * sizeof(rvec), pmeGPU->archSpecific->pmeStream);
}

void pme_gpu_free_coordinates(const pme_gpu_t *pmeGPU)
{
    cu_free_buffered(pmeGPU->kernelParams.get()->atoms.d_coordinates, &pmeGPU->archSpecific->coordinatesSize, &pmeGPU->archSpecific->coordinatesSizeAlloc);
}

void pme_gpu_realloc_and_copy_input_coefficients(const pme_gpu_t *pmeGPU, const float *h_coefficients)
{
    GMX_ASSERT(h_coefficients, "Bad host-side charge buffer in PME GPU");
    const size_t newCoefficientsSize = pmeGPU->nAtomsAlloc;
    GMX_ASSERT(newCoefficientsSize > 0, "Bad number of atoms in PME GPU");
    cu_realloc_buffered((void **)&pmeGPU->kernelParams.get()->atoms.d_coefficients, NULL, sizeof(float),
                        &pmeGPU->archSpecific->coefficientsSize, &pmeGPU->archSpecific->coefficientsSizeAlloc,
                        newCoefficientsSize, pmeGPU->archSpecific->pmeStream, true);
    pme_gpu_make_sure_memory_is_pinned((void **)&h_coefficients, pmeGPU->kernelParams.get()->atoms.nAtoms * sizeof(float));
    cu_copy_H2D_async(pmeGPU->kernelParams.get()->atoms.d_coefficients, const_cast<float *>(h_coefficients),
                      pmeGPU->kernelParams.get()->atoms.nAtoms * sizeof(float), pmeGPU->archSpecific->pmeStream);
#if PME_GPU_USE_PADDING
    const size_t paddingIndex = pmeGPU->kernelParams.get()->atoms.nAtoms;
    const size_t paddingCount = pmeGPU->nAtomsAlloc - paddingIndex;
    if (paddingCount > 0)
    {
        cudaError_t stat = cudaMemsetAsync(pmeGPU->kernelParams.get()->atoms.d_coefficients + paddingIndex, 0, paddingCount * sizeof(float), pmeGPU->archSpecific->pmeStream);
        CU_RET_ERR(stat, "PME failed to clear the padded charges");
    }
#endif
}

void pme_gpu_free_coefficients(const pme_gpu_t *pmeGPU)
{
    cu_free_buffered(pmeGPU->kernelParams.get()->atoms.d_coefficients, &pmeGPU->archSpecific->coefficientsSize, &pmeGPU->archSpecific->coefficientsSizeAlloc);
}

void pme_gpu_realloc_spline_data(const pme_gpu_t *pmeGPU)
{
    const int    order             = pmeGPU->common->pme_order;
    const int    alignment         = PME_SPREADGATHER_ATOMS_PER_WARP;
    const size_t nAtomsPadded      = ((pmeGPU->nAtomsAlloc + alignment - 1) / alignment) * alignment;
    const size_t newSplineDataSize = DIM * order * nAtomsPadded;
    GMX_ASSERT(newSplineDataSize > 0, "Bad number of atoms in PME GPU");
    /* Two arrays of the same size */
    int currentSizeTemp      = pmeGPU->archSpecific->splineDataSize;
    int currentSizeTempAlloc = pmeGPU->archSpecific->splineDataSizeAlloc;
    cu_realloc_buffered((void **)&pmeGPU->kernelParams.get()->atoms.d_theta, NULL, sizeof(float),
                        &currentSizeTemp, &currentSizeTempAlloc, newSplineDataSize, pmeGPU->archSpecific->pmeStream, true);
    cu_realloc_buffered((void **)&pmeGPU->kernelParams.get()->atoms.d_dtheta, NULL, sizeof(float),
                        &pmeGPU->archSpecific->splineDataSize, &pmeGPU->archSpecific->splineDataSizeAlloc, newSplineDataSize, pmeGPU->archSpecific->pmeStream, true);
}

void pme_gpu_free_spline_data(const pme_gpu_t *pmeGPU)
{
    /* Two arrays of the same size */
    cu_free_buffered(pmeGPU->kernelParams.get()->atoms.d_theta);
    cu_free_buffered(pmeGPU->kernelParams.get()->atoms.d_dtheta, &pmeGPU->archSpecific->splineDataSize, &pmeGPU->archSpecific->splineDataSizeAlloc);
}

void pme_gpu_realloc_grid_indices(const pme_gpu_t *pmeGPU)
{
    const size_t newIndicesSize = DIM * pmeGPU->nAtomsAlloc;
    GMX_ASSERT(newIndicesSize > 0, "Bad number of atoms in PME GPU");
    cu_realloc_buffered((void **)&pmeGPU->kernelParams.get()->atoms.d_gridlineIndices, NULL, sizeof(int),
                        &pmeGPU->archSpecific->gridlineIndicesSize, &pmeGPU->archSpecific->gridlineIndicesSizeAlloc, newIndicesSize, pmeGPU->archSpecific->pmeStream, true);
}

void pme_gpu_free_grid_indices(const pme_gpu_t *pmeGPU)
{
    cu_free_buffered(pmeGPU->kernelParams.get()->atoms.d_gridlineIndices, &pmeGPU->archSpecific->gridlineIndicesSize, &pmeGPU->archSpecific->gridlineIndicesSizeAlloc);
}

void pme_gpu_realloc_grids(pme_gpu_t *pmeGPU)
{
    // TODO: make tests to be assured this grid size is always suffcieint for copying the CPU grids
    // TODO: put the gridsize in the structure maybe?
    pme_gpu_cuda_kernel_params_t *kernelParamsPtr = pmeGPU->kernelParams.get();
    const int                     newGridSize     = kernelParamsPtr->grid.localGridSizePadded[XX] *
        kernelParamsPtr->grid.localGridSizePadded[YY] *
        kernelParamsPtr->grid.localGridSizePadded[ZZ];

    if (pmeGPU->archSpecific->performOutOfPlaceFFT)
    {
        /* Allocate a separate complex grid */
        int tempGridSize      = pmeGPU->archSpecific->gridSize;
        int tempGridSizeAlloc = pmeGPU->archSpecific->gridSizeAlloc;
        cu_realloc_buffered((void **)&kernelParamsPtr->grid.d_fourierGrid, NULL, sizeof(float),
                            &tempGridSize, &tempGridSizeAlloc, newGridSize, pmeGPU->archSpecific->pmeStream, true);
    }
    cu_realloc_buffered((void **)&kernelParamsPtr->grid.d_realGrid, NULL, sizeof(float),
                        &pmeGPU->archSpecific->gridSize, &pmeGPU->archSpecific->gridSizeAlloc, newGridSize, pmeGPU->archSpecific->pmeStream, true);
    if (!pmeGPU->archSpecific->performOutOfPlaceFFT)
    {
        /* Using the same grid */
        kernelParamsPtr->grid.d_fourierGrid = kernelParamsPtr->grid.d_realGrid;
    }
}

void pme_gpu_free_grids(const pme_gpu_t *pmeGPU)
{
    if (pmeGPU->archSpecific->performOutOfPlaceFFT)
    {
        /* Free a separate complex grid of the same size */
        cu_free_buffered(pmeGPU->kernelParams.get()->grid.d_fourierGrid);
    }
    cu_free_buffered(pmeGPU->kernelParams.get()->grid.d_realGrid, &pmeGPU->archSpecific->gridSize, &pmeGPU->archSpecific->gridSizeAlloc);
}

void pme_gpu_clear_grids(const pme_gpu_t *pmeGPU)
{
    cudaError_t stat = cudaMemsetAsync(pmeGPU->kernelParams.get()->grid.d_realGrid, 0,
                                       pmeGPU->archSpecific->gridSize * sizeof(float), pmeGPU->archSpecific->pmeStream);
    /* Should the complex grid be cleared in some weird case? */
    CU_RET_ERR(stat, "cudaMemsetAsync on the PME grid error");
}

void pme_gpu_realloc_and_copy_fract_shifts(pme_gpu_t *pmeGPU)
{
    cudaStream_t                  s               = pmeGPU->archSpecific->pmeStream;
    pme_gpu_cuda_kernel_params_t *kernelParamsPtr = pmeGPU->kernelParams.get();

    const int                     nx = kernelParamsPtr->grid.localGridSize[XX];
    const int                     ny = kernelParamsPtr->grid.localGridSize[YY];
    const int                     nz = kernelParamsPtr->grid.localGridSize[ZZ];

    const int                     cellCount = c_pmeNeighborUnitcellCount;

    const int                     fshOffset[DIM] = {0, cellCount * nx, cellCount * (nx + ny)};
    memcpy(kernelParamsPtr->grid.tablesOffsets, &fshOffset, sizeof(fshOffset));

    const int    newFractShiftsSize  = cellCount * (nx + ny + nz);

    /* Two arrays, same size */
    int currentSizeTemp      = pmeGPU->archSpecific->fractShiftsSize;
    int currentSizeTempAlloc = pmeGPU->archSpecific->fractShiftsSizeAlloc;
    cu_realloc_buffered((void **)&kernelParamsPtr->grid.d_fractShiftsTable, NULL, sizeof(float),
                        &currentSizeTemp, &currentSizeTempAlloc,
                        newFractShiftsSize, pmeGPU->archSpecific->pmeStream, true);
    float *fshArray = kernelParamsPtr->grid.d_fractShiftsTable;
    cu_realloc_buffered((void **)&kernelParamsPtr->grid.d_gridlineIndicesTable, NULL, sizeof(int),
                        &pmeGPU->archSpecific->fractShiftsSize, &pmeGPU->archSpecific->fractShiftsSizeAlloc,
                        newFractShiftsSize, pmeGPU->archSpecific->pmeStream, true);
    int *nnArray = kernelParamsPtr->grid.d_gridlineIndicesTable;

    /* TODO: pinning */

    for (int i = 0; i < DIM; i++)
    {
        kernelParamsPtr->grid.tablesOffsets[i] = fshOffset[i];
        cu_copy_H2D_async(fshArray + fshOffset[i], pmeGPU->common->fsh[i].data(), cellCount * kernelParamsPtr->grid.localGridSize[i] * sizeof(float), s);
        cu_copy_H2D_async(nnArray + fshOffset[i], pmeGPU->common->nn[i].data(), cellCount * kernelParamsPtr->grid.localGridSize[i] * sizeof(int), s);
    }

    /* TODO: fix the textures code */
    pme_gpu_make_fract_shifts_textures(pmeGPU);
}

void pme_gpu_free_fract_shifts(const pme_gpu_t *pmeGPU)
{
    pme_gpu_free_fract_shifts_textures(pmeGPU);

    /* Two arrays, same size */
    cu_free_buffered(pmeGPU->kernelParams.get()->grid.d_fractShiftsTable);
    cu_free_buffered(pmeGPU->kernelParams.get()->grid.d_gridlineIndicesTable, &pmeGPU->archSpecific->fractShiftsSize, &pmeGPU->archSpecific->fractShiftsSizeAlloc);
}

void pme_gpu_sync_output_energy_virial(const pme_gpu_t *pmeGPU)
{
    cudaError_t stat = cudaStreamWaitEvent(pmeGPU->archSpecific->pmeStream, pmeGPU->archSpecific->syncEnerVirD2H, 0);
    CU_RET_ERR(stat, "Error while waiting for PME solve");

    for (int j = 0; j < PME_GPU_VIRIAL_AND_ENERGY_COUNT; j++)
    {
        GMX_ASSERT(!isnan(pmeGPU->staging.h_virialAndEnergy[j]), "PME GPU produces incorrect energy/virial.");
    }
}

void pme_gpu_sync_grid(const pme_gpu_t *pmeGPU, const gmx_fft_direction dir)
{
    /* FIXME: this function does not actually seem to be used when it should be, with CPU FFT? */
    bool syncGPUGrid = ((dir == GMX_FFT_REAL_TO_COMPLEX) ? true : pme_gpu_performs_solve(pmeGPU));
    if (syncGPUGrid)
    {
        cudaEvent_t syncEvent = (dir == GMX_FFT_REAL_TO_COMPLEX) ? pmeGPU->archSpecific->syncSpreadGridD2H : pmeGPU->archSpecific->syncSolveGridD2H;
        cudaError_t stat      = cudaStreamWaitEvent(pmeGPU->archSpecific->pmeStream, syncEvent, 0);
        CU_RET_ERR(stat, "Error while waiting for the PME GPU grid to be copied to CPU");
    }
}

void pme_gpu_init_specific(pme_gpu_t *pmeGPU, const gmx_hw_info_t *hwinfo, const gmx_gpu_opt_t *gpu_opt)
{
    /* FIXME: fix the GPU ID selection as well as initialization */
    int       gpuIndex = 0;
    char      gpu_err_str[STRLEN];
    GMX_RELEASE_ASSERT(hwinfo, "No hardware information");
    GMX_RELEASE_ASSERT(hwinfo->gpu_info.gpu_dev, "No GPU information");
    GMX_RELEASE_ASSERT(gpu_opt->dev_use, "No GPU information");
    /* Use GPU #0 for now since the code for GPU init has to be reworked anyway.
     * And don't forget to resurrect the external GMX_PME_GPU_ID env. variable.
     */
    pmeGPU->deviceInfo = &hwinfo->gpu_info.gpu_dev[gpu_opt->dev_use[gpuIndex]];
    const gmx::MDLogger temp;
    if (!init_gpu(temp, gpuIndex, gpu_err_str, &hwinfo->gpu_info, gpu_opt))
    {
        gmx_fatal(FARGS, "Could not select GPU %d for PME rank %d\n", pmeGPU->deviceInfo->id, gpuIndex);
    }

    /* Allocate the GPU-specific structures */
    pmeGPU->archSpecific.reset(new pme_gpu_specific_host_t());
    pmeGPU->kernelParams.reset(new pme_gpu_kernel_params_t());

    pmeGPU->archSpecific->performOutOfPlaceFFT = true;
    /* This should give better performance, according to the cuFFT documentation.
     * The performance seems to be the same though.
     * Perhaps the limiting factor is using paddings/overlaps in the grid, which is also frowned upon.
     * PME could also try to pick up nice grid sizes (with factors of 2, 3, 5, 7).
     */

    pmeGPU->archSpecific->useTiming = (getenv("GMX_DISABLE_CUDA_TIMING") == NULL);
    /* FIXME: This should also check for NB GPU being launched, and NB should check for PME GPU!
     * Multiple CUDA streams on same GPU cause nonsense cudaEvent_t timings.
     */

    //pmeGPU->archSpecific->bUseTextureObjects = (pmeGPU->deviceInfo->prop.major >= 3);
    /* TODO: have to fix the GPU id selection */

    /* Creating a PME CUDA stream */
    cudaError_t stat;
    int         highest_priority, lowest_priority;
    stat = cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
    CU_RET_ERR(stat, "PME cudaDeviceGetStreamPriorityRange failed");
    stat = cudaStreamCreateWithPriority(&pmeGPU->archSpecific->pmeStream,
                                        cudaStreamDefault, //cudaStreamNonBlocking,
                                        highest_priority);
    CU_RET_ERR(stat, "cudaStreamCreateWithPriority on the PME stream failed");
}

void pme_gpu_destroy_specific(const pme_gpu_t *pmeGPU)
{
    /* Destroy the CUDA stream */
    cudaError_t stat = cudaStreamDestroy(pmeGPU->archSpecific->pmeStream);
    CU_RET_ERR(stat, "PME cudaStreamDestroy error");
}

void pme_gpu_init_sync_events(const pme_gpu_t *pmeGPU)
{
    cudaError_t stat;
    stat = cudaEventCreateWithFlags(&pmeGPU->archSpecific->syncEnerVirD2H, cudaEventDisableTiming);
    CU_RET_ERR(stat, "cudaEventCreate on syncEnerVirH2D failed");
    stat = cudaEventCreateWithFlags(&pmeGPU->archSpecific->syncForcesD2H, cudaEventDisableTiming);
    CU_RET_ERR(stat, "cudaEventCreate on syncForcesH2D failed");
    stat = cudaEventCreateWithFlags(&pmeGPU->archSpecific->syncSpreadGridD2H, cudaEventDisableTiming);
    CU_RET_ERR(stat, "cudaEventCreate on syncSpreadGridH2D failed");
    stat = cudaEventCreateWithFlags(&pmeGPU->archSpecific->syncSolveGridD2H, cudaEventDisableTiming);
    CU_RET_ERR(stat, "cudaEventCreate on syncSolveGridH2D failed");
}

void pme_gpu_destroy_sync_events(const pme_gpu_t *pmeGPU)
{
    cudaError_t stat;
    stat = cudaEventDestroy(pmeGPU->archSpecific->syncEnerVirD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncEnerVirH2D");
    stat = cudaEventDestroy(pmeGPU->archSpecific->syncForcesD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncForcesH2D");
    stat = cudaEventDestroy(pmeGPU->archSpecific->syncSpreadGridD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncpreadGridH2D");
    stat = cudaEventDestroy(pmeGPU->archSpecific->syncSolveGridD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncSolveGridH2D");
}

#if PME_GPU_PAGELOCKING_HACK
#include <set>
static std::set<void *> pageLockedPointers;
#endif

void pme_gpu_make_sure_memory_is_pinned(void **h_ptr, size_t bytes)
{
#if PME_GPU_PAGELOCKING_HACK
    if (!pageLockedPointers.count(*h_ptr))
    {
        cudaError_t stat = cudaHostRegister(*h_ptr, bytes, cudaHostRegisterDefault);
        if (stat == cudaErrorHostMemoryAlreadyRegistered)
        {
            cudaGetLastError(); // suppress "Already mapped" message
        }
        else
        {
            CU_RET_ERR(stat, "Could not pin the PME GPU memory");
        }
        pageLockedPointers.insert(*h_ptr);
    }
#else
    GMX_UNUSED_VALUE(h_ptr);
    GMX_UNUSED_VALUE(bytes);
#endif
}

void pme_gpu_reinit_3dfft(const pme_gpu_t *pmeGPU)
{
    if (pme_gpu_performs_FFT(pmeGPU))
    {
        pmeGPU->archSpecific->pfft_setup_gpu.resize(0); // FIXME: reallocations
        for (int i = 0; i < pmeGPU->common->ngrids; i++)
        {
            pmeGPU->archSpecific->pfft_setup_gpu.push_back(std::unique_ptr<parallel_3dfft_gpu_t>(new parallel_3dfft_gpu_t(pmeGPU)));
        }
    }
}

void pme_gpu_destroy_3dfft(const pme_gpu_t *pmeGPU)
{
    pmeGPU->archSpecific->pfft_setup_gpu.resize(0);
}
