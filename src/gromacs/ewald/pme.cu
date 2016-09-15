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
 *  \brief Implements common PME GPU routines in CUDA.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

/* GPU initialization includes */
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/utility/logger.h"

/* The rest */
#include <assert.h>
#include "gromacs/gpu_utils/pmalloc_cuda.h"
#include "gromacs/math/invertmatrix.h"
#include "gromacs/math/units.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxomp.h"
#include "gromacs/utility/smalloc.h"
#include "pme.cuh"
#include "pme.h"
#include "pme-grid.h"
#include "pme-solve.h"

/*! \brief \internal
 * Allocates the fixed size energy and virial buffer both on GPU and CPU.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_alloc_energy_virial(const gmx_pme_t *pme)
{
    const size_t energyAndVirialSize = PME_GPU_VIRIAL_AND_ENERGY_COUNT * sizeof(float);
    cudaError_t  stat                = cudaMalloc((void **)&pme->gpu->kernelParams.constants.virialAndEnergy, energyAndVirialSize);
    CU_RET_ERR(stat, "cudaMalloc failed on PME energy and virial");
    pmalloc((void **)&pme->gpu->virialAndEnergyHost, energyAndVirialSize);
}

/*! \brief \internal
 * Frees the energy and virial memory both on GPU and CPU.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_energy_virial(const gmx_pme_t *pme)
{
    cudaError_t stat = cudaFree(pme->gpu->kernelParams.constants.virialAndEnergy);
    CU_RET_ERR(stat, "cudaFree failed on PME energy and virial");
    pme->gpu->kernelParams.constants.virialAndEnergy = NULL;
    pfree(pme->gpu->virialAndEnergyHost);
    pme->gpu->virialAndEnergyHost = NULL;
}

/*! \brief \internal
 *
 * Clears the energy and virial memory on GPU with 0.
 * Should be called at the end of the energy/virial calculation step.
 */
void pme_gpu_clear_energy_virial(const gmx_pme_t *pme)
{
    cudaError_t stat = cudaMemsetAsync(pme->gpu->kernelParams.constants.virialAndEnergy, 0,
                                       PME_GPU_VIRIAL_AND_ENERGY_COUNT * sizeof(float), pme->gpu->pmeStream);
    CU_RET_ERR(stat, "PME energies/virial cudaMemsetAsync error");
}

void pme_gpu_get_energy_virial(const gmx_pme_t *pme, real *energy, matrix virial)
{
    assert(energy);
    size_t j = 0;
    virial[XX][XX] = 0.25 * pme->gpu->virialAndEnergyHost[j++];
    virial[YY][YY] = 0.25 * pme->gpu->virialAndEnergyHost[j++];
    virial[ZZ][ZZ] = 0.25 * pme->gpu->virialAndEnergyHost[j++];
    virial[XX][YY] = virial[YY][XX] = 0.25 * pme->gpu->virialAndEnergyHost[j++];
    virial[XX][ZZ] = virial[ZZ][XX] = 0.25 * pme->gpu->virialAndEnergyHost[j++];
    virial[YY][ZZ] = virial[ZZ][YY] = 0.25 * pme->gpu->virialAndEnergyHost[j++];
    *energy        = 0.5 * pme->gpu->virialAndEnergyHost[j++];
}

/*! \brief \internal
 *
 * Reallocates and copies the pre-computed B-spline values to the GPU.
 * FIXME: currently uses just a global memory, could be using texture memory/ldg.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_realloc_and_copy_bspline_values(const gmx_pme_t *pme)
{
    const int splineValuesOffset[DIM] = {0, pme->nkx, pme->nkx + pme->nky}; //?replace nkx
    memcpy(&pme->gpu->kernelParams.grid.splineValuesOffset, &splineValuesOffset, sizeof(splineValuesOffset));

    const int newSplineValuesSize = pme->nkx + pme->nky + pme->nkz;
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.grid.splineValuesArray, NULL, sizeof(float),
                        &pme->gpu->splineValuesSize, &pme->gpu->splineValuesSizeAlloc, newSplineValuesSize, pme->gpu->pmeStream, true);

    for (int i = 0; i < DIM; i++)
    {
        size_t       gridSize;
        switch (i)
        {
            case XX:
                gridSize = pme->nkx;
                break;

            case YY:
                gridSize = pme->nky;
                break;

            case ZZ:
                gridSize = pme->nkz;
                break;
        }
        size_t  modSize  = gridSize * sizeof(float);
        /* reallocate the host buffer */
        if ((pme->gpu->splineValuesHost[i] == NULL) || (pme->gpu->splineValuesHostSizes[i] < modSize))
        {
            pfree(pme->gpu->splineValuesHost[i]);
            pmalloc((void **)&pme->gpu->splineValuesHost[i], modSize);
        }
        memcpy(pme->gpu->splineValuesHost[i], pme->bsp_mod[i], modSize);
        //yupinov instead use pinning here as well!
        cu_copy_H2D_async(pme->gpu->kernelParams.grid.splineValuesArray + splineValuesOffset[i], pme->gpu->splineValuesHost[i], modSize, pme->gpu->pmeStream);
    }
}

/*! \brief \internal
 * Frees the pre-computed B-spline values on the GPU (and the transfer CPU buffers).
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_bspline_values(const gmx_pme_t *pme)
{
    for (int i = 0; i < DIM; i++)
    {
        pfree(pme->gpu->splineValuesHost[i]);
    }
    cu_free_buffered(pme->gpu->kernelParams.grid.splineValuesArray, &pme->gpu->splineValuesSize, &pme->gpu->splineValuesSizeAlloc);
}

/*! \brief \internal
 * Copies the grid sizes for overlapping (used in the PME wrap/unwrap).
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_copy_wrap_zones(const gmx_pme_t *pme)
{
    const int nx      = pme->gpu->kernelParams.grid.localGridSize.x;
    const int ny      = pme->gpu->kernelParams.grid.localGridSize.y;
    const int nz      = pme->gpu->kernelParams.grid.localGridSize.z;
    const int overlap = pme->pme_order - 1;

    /* Cell counts in the 7 overlapped grid parts */
    /* Is this correct? No Z alignment changes? */
    const int3 zoneSizes_h[PME_GPU_OVERLAP_ZONES_COUNT] =
    {
        {     nx,        ny,   overlap},
        {     nx,   overlap,        nz},
        {overlap,        ny,        nz},
        {     nx,   overlap,   overlap},
        {overlap,        ny,   overlap},
        {overlap,   overlap,        nz},
        {overlap,   overlap,   overlap}
    };
    /* The X is never used on the GPU, actually */
    int2 zoneSizesYZ_h[PME_GPU_OVERLAP_ZONES_COUNT];
    for (int i = 0; i < PME_GPU_OVERLAP_ZONES_COUNT; i++)
    {
        zoneSizesYZ_h[i].x = zoneSizes_h[i].y;
        zoneSizesYZ_h[i].y = zoneSizes_h[i].z;
    }
    int cellsAccumCount_h[PME_GPU_OVERLAP_ZONES_COUNT];
    for (int i = 0; i < PME_GPU_OVERLAP_ZONES_COUNT; i++)
    {
        cellsAccumCount_h[i] = zoneSizes_h[i].x * zoneSizes_h[i].y * zoneSizes_h[i].z;
    }
    /* Accumulation */
    for (int i = 1; i < PME_GPU_OVERLAP_ZONES_COUNT; i++)
    {
        cellsAccumCount_h[i] = cellsAccumCount_h[i] + cellsAccumCount_h[i - 1];
    }
    memcpy(pme->gpu->kernelParams.grid.overlapSizes, zoneSizesYZ_h, sizeof(zoneSizesYZ_h));
    memcpy(pme->gpu->kernelParams.grid.overlapCellCounts, cellsAccumCount_h, sizeof(cellsAccumCount_h));
}

/*! \brief \internal
 * Reallocates the GPU buffer for the PME forces.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_realloc_forces(const gmx_pme_t *pme)
{
    const size_t newForcesSize = pme->gpu->nAtomsAlloc * DIM;
    assert(newForcesSize > 0);
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.atoms.forces, NULL, sizeof(float),
                        &pme->gpu->forcesSize, &pme->gpu->forcesSizeAlloc, newForcesSize, pme->gpu->pmeStream, true);
}

/*! \brief \internal
 * Frees the GPU buffer for the PME forces.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_forces(const gmx_pme_t *pme)
{
    cu_free_buffered(pme->gpu->kernelParams.atoms.forces, &pme->gpu->forcesSize, &pme->gpu->forcesSizeAlloc);
}

/*! \brief \internal
 * Reallocates the input coordinates buffer on the GPU (and clears the padded part if needed).
 *
 * \param[in] pme            The PME structure.
 *
 * Needs to be called on every DD step/in the beginning.
 */
void pme_gpu_realloc_coordinates(const gmx_pme_t *pme)
{
    const size_t newCoordinatesSize = pme->gpu->nAtomsAlloc * DIM;
    assert(newCoordinatesSize > 0);
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.atoms.coordinates, NULL, sizeof(float),
                        &pme->gpu->coordinatesSize, &pme->gpu->coordinatesSizeAlloc, newCoordinatesSize, pme->gpu->pmeStream, true);
#if PME_GPU_USE_PADDING
    const size_t paddingIndex = DIM * pme->gpu->kernelParams.atoms.nAtoms;
    const size_t paddingCount = DIM * pme->gpu->nAtomsAlloc - paddingIndex;
    if (paddingCount > 0)
    {
        cudaError_t stat = cudaMemsetAsync(pme->gpu->kernelParams.atoms.coordinates + paddingIndex, 0, paddingCount * sizeof(float), pme->gpu->pmeStream);
        CU_RET_ERR(stat, "PME failed to clear the padded coordinates");
    }
#endif
}

/*! \brief \internal
 * Copies the input coordinates from the CPU buffer (pme->gpu->coordinatesHost) onto the GPU.
 *
 * \param[in] pme            The PME structure.
 *
 * Needs to be called every MD step. The coordinates are then used in the spline calculation.
 */
void pme_gpu_copy_coordinates(const gmx_pme_t *pme)
{
    assert(pme->gpu->coordinatesHost);
    cu_copy_H2D_async(pme->gpu->kernelParams.atoms.coordinates, pme->gpu->coordinatesHost, pme->gpu->kernelParams.atoms.nAtoms * DIM * sizeof(float), pme->gpu->pmeStream);
}

/*! \brief \internal
 * Frees the coordinates on the GPU.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_coordinates(const gmx_pme_t *pme)
{
    cu_free_buffered(pme->gpu->kernelParams.atoms.coordinates, &pme->gpu->coordinatesSize, &pme->gpu->coordinatesSizeAlloc);
}

/*! \brief \internal
 * Reallocates the buffer on the GPU and copies the charges/coefficients from the CPU buffer (pme->gpu->coefficientsHost). Clears the padded part if needed.
 *
 * \param[in] pme            The PME structure.
 *
 * Does not need to be done every MD step, only whenever the local charges change.
 * (So, in the beginning of the run, or on DD step).
 */
void pme_gpu_realloc_and_copy_coefficients(const gmx_pme_t *pme)
{
    assert(pme->gpu->coefficientsHost);
    const size_t newCoefficientsSize = pme->gpu->nAtomsAlloc;
    assert(newCoefficientsSize > 0);
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.atoms.coefficients, NULL, sizeof(float),
                        &pme->gpu->coefficientsSize, &pme->gpu->coefficientsSizeAlloc, newCoefficientsSize, pme->gpu->pmeStream, true);
    cu_copy_H2D_async(pme->gpu->kernelParams.atoms.coefficients, pme->gpu->coefficientsHost, pme->gpu->kernelParams.atoms.nAtoms * sizeof(float), pme->gpu->pmeStream);
#if PME_GPU_USE_PADDING
    const size_t paddingIndex = pme->gpu->kernelParams.atoms.nAtoms;
    const size_t paddingCount = pme->gpu->nAtomsAlloc - paddingIndex;
    if (paddingCount > 0)
    {
        cudaError_t stat = cudaMemsetAsync(pme->gpu->kernelParams.atoms.coefficients + paddingIndex, 0, paddingCount * sizeof(float), pme->gpu->pmeStream);
        CU_RET_ERR(stat, "PME failed to clear the padded charges");
    }
#endif
}

/*! \brief \internal
 * Frees the charges on the GPU.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_charges(const gmx_pme_t *pme)
{
    cu_free_buffered(pme->gpu->kernelParams.atoms.coefficients, &pme->gpu->coefficientsSize, &pme->gpu->coefficientsSizeAlloc);
}

/*! \brief \internal
 * Reallocates the buffers on the GPU for the atoms spline data.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_realloc_spline_data(const gmx_pme_t *pme)
{
    const int    order             = pme->pme_order;
    const int    alignment         = PME_SPREADGATHER_PARTICLES_PER_WARP;
    const size_t nAtomsPadded      = ((pme->gpu->nAtomsAlloc + alignment - 1) / alignment) * alignment;
    const size_t newSplineDataSize = DIM * order * nAtomsPadded;
    assert(newSplineDataSize > 0);
    /* Two arrays of the same size */
    int currentSizeTemp      = pme->gpu->splineDataSize;
    int currentSizeTempAlloc = pme->gpu->splineDataSizeAlloc;
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.atoms.theta, NULL, sizeof(float),
                        &currentSizeTemp, &currentSizeTempAlloc, newSplineDataSize, pme->gpu->pmeStream, true);
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.atoms.dtheta, NULL, sizeof(float),
                        &pme->gpu->splineDataSize, &pme->gpu->splineDataSizeAlloc, newSplineDataSize, pme->gpu->pmeStream, true);
}

/*! \brief \internal
 * Frees the buffers on the GPU for the atoms spline data.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_spline_data(const gmx_pme_t *pme)
{
    /* Two arrays of the same size */
    cu_free_buffered(pme->gpu->kernelParams.atoms.theta);
    cu_free_buffered(pme->gpu->kernelParams.atoms.dtheta, &pme->gpu->splineDataSize, &pme->gpu->splineDataSizeAlloc);
}

/*! \brief \internal
 * Reallocates the buffer on the GPU for the particle gridline indices.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_realloc_grid_indices(const gmx_pme_t *pme)
{
    const size_t newIndicesSize = DIM * pme->gpu->nAtomsAlloc;
    assert(newIndicesSize > 0);
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.atoms.gridlineIndices, NULL, sizeof(int),
                        &pme->gpu->gridlineIndicesSize, &pme->gpu->gridlineIndicesSizeAlloc, newIndicesSize, pme->gpu->pmeStream, true);
}

/*! \brief
 * Frees the buffer on the GPU for the particle gridline indices.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_grid_indices(const gmx_pme_t *pme)
{
    cu_free_buffered(pme->gpu->kernelParams.atoms.gridlineIndices, &pme->gpu->gridlineIndicesSize, &pme->gpu->gridlineIndicesSizeAlloc);
}

void pme_gpu_realloc_grids(const gmx_pme_t *pme)
{
    const int pnx         = pme->pmegrid_nx; //?
    const int pny         = pme->pmegrid_ny;
    const int pnz         = pme->pmegrid_nz;
    const int newGridSize = pnx * pny * pnz;

    if (pme->gpu->bOutOfPlaceFFT)
    {
        /* Allocate a separate complex grid */
        int tempGridSize      = pme->gpu->gridSize;
        int tempGridSizeAlloc = pme->gpu->gridSizeAlloc;
        cu_realloc_buffered((void **)&pme->gpu->kernelParams.grid.fourierGrid, NULL, sizeof(float),
                            &tempGridSize, &tempGridSizeAlloc, newGridSize, pme->gpu->pmeStream, true);
    }
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.grid.realGrid, NULL, sizeof(float),
                        &pme->gpu->gridSize, &pme->gpu->gridSizeAlloc, newGridSize, pme->gpu->pmeStream, true);
    if (!pme->gpu->bOutOfPlaceFFT)
    {
        /* Using the same grid */
        pme->gpu->kernelParams.grid.fourierGrid = (float2 *)(pme->gpu->kernelParams.grid.realGrid);
    }
}

void pme_gpu_free_grids(const gmx_pme_t *pme)
{
    if (pme->gpu->bOutOfPlaceFFT)
    {
        /* Free a separate complex grid of the same size */
        cu_free_buffered(pme->gpu->kernelParams.grid.fourierGrid);
    }
    cu_free_buffered(pme->gpu->kernelParams.grid.realGrid, &pme->gpu->gridSize, &pme->gpu->gridSizeAlloc);
}

void pme_gpu_clear_grids(const gmx_pme_t *pme)
{
    cudaError_t stat = cudaMemsetAsync(pme->gpu->kernelParams.grid.realGrid, 0, pme->gpu->gridSize * sizeof(float), pme->gpu->pmeStream);
    /* Should the complex grid be cleared in some weird case? */
    CU_RET_ERR(stat, "cudaMemsetAsync on the PME grid error");
}

/*! \brief \internal
 * The PME GPU reinitialization function that is called both at the end of any MD step and on any load balancing step.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_step_reinit(const gmx_pme_t *pme)
{
    pme_gpu_clear_grids(pme);
    pme_gpu_clear_energy_virial(pme);
}

void pme_gpu_init(gmx_pme_t *pme, const gmx_hw_info_t *hwinfo, const gmx_gpu_opt_t *gpu_opt)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    const gmx_bool firstInit = !pme->gpu;
    if (firstInit)
    {
        snew(pme->gpu, 1);
        cudaError_t stat;

        /* GPU selection copied from non-bondeds */
        const int PMEGPURank = pme->nodeid;
        char      gpu_err_str[STRLEN];
        assert(hwinfo);
        assert(hwinfo->gpu_info.gpu_dev);
        assert(gpu_opt->dev_use);

        int   forcedGPUId       = -1;
        char *forcedGPUIdString = getenv("GMX_PME_GPU_ID");
        if (forcedGPUIdString)
        {
            forcedGPUId = atoi(forcedGPUIdString);
            printf("PME rank %d trying to use GPU %d\n", PMEGPURank, forcedGPUId);
            stat = cudaSetDevice(forcedGPUId);
            CU_RET_ERR(stat, "PME failed to set the GPU device");
        }
        else
        {
            pme->gpu->deviceInfo = &hwinfo->gpu_info.gpu_dev[gpu_opt->dev_use[PMEGPURank]];
            const gmx::MDLogger temp;
            if (!init_gpu(temp, PMEGPURank, gpu_err_str, &hwinfo->gpu_info, gpu_opt))
            {
                gmx_fatal(FARGS, "Could not select GPU %d for PME rank %d\n", pme->gpu->deviceInfo->id, PMEGPURank);
            }
        }

        /* Some permanent settings are set here */

        pme->gpu->bGPUSingle = pme_gpu_enabled(pme) && (pme->nnodes == 1);
        /* A convenience variable. */

        pme->gpu->bGPUFFT = pme->gpu->bGPUSingle && !getenv("GMX_PME_GPU_FFTW");
        /* cuFFT will only used for a single rank. */

        pme->gpu->bGPUSolve = TRUE;
        /* pme->gpu->bGPUFFT - CPU solve with the CPU FFTW is definitely broken at the moment - 20160511 */

        pme->gpu->bGPUGather = TRUE;
        /* CPU gather has got to be broken as well due to different theta/dtheta layout. */

        pme->gpu->bOutOfPlaceFFT = TRUE;
        /* This should give better performance, according to the cuFFT documentation.
         * The performance seems to be the same though.
         * Perhaps the limiting factor is using paddings/overlaps in the grid, which is also frowned upon.
         * PME could also try to pick up nice grid sizes (with factors of 2, 3, 5, 7)
         */

        pme->gpu->bNeedToUpdateAtoms = TRUE;                             /* For the delayed atom data init */

        pme->gpu->bTiming = (getenv("GMX_DISABLE_CUDA_TIMING") == NULL); /* This should also check for NB GPU being launched, and NB should check for PME GPU! */

        //pme->gpu->bUseTextureObjects = (pme->gpu->deviceInfo->prop.major >= 3);
        //yupinov - have to fix the GPU id selection, forced GPUIdHack?

        /* Creating a PME CUDA stream */
#if GMX_CUDA_VERSION >= 5050
        int highest_priority;
        int lowest_priority;
        stat = cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
        CU_RET_ERR(stat, "PME cudaDeviceGetStreamPriorityRange failed");
        stat = cudaStreamCreateWithPriority(&pme->gpu->pmeStream,
                                            cudaStreamDefault, //cudaStreamNonBlocking,
                                            highest_priority);

        CU_RET_ERR(stat, "cudaStreamCreateWithPriority on the PME stream failed");
#else
        stat = cudaStreamCreate(&pme->gpu->pmeStream);
        CU_RET_ERR(stat, "PME cudaStreamCreate error");
#endif

        /* Creating synchronization events */
        stat = cudaEventCreateWithFlags(&pme->gpu->syncEnerVirD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncEnerVirH2D failed");
        stat = cudaEventCreateWithFlags(&pme->gpu->syncForcesD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncForcesH2D failed");
        stat = cudaEventCreateWithFlags(&pme->gpu->syncSpreadGridD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncSpreadGridH2D failed");
        stat = cudaEventCreateWithFlags(&pme->gpu->syncSolveGridD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncSolveGridH2D failed");

        pme_gpu_init_timings(pme);

        pme_gpu_alloc_energy_virial(pme);

        assert(pme->epsilon_r != 0.0f);
        pme->gpu->kernelParams.constants.elFactor = ONE_4PI_EPS0 / pme->epsilon_r;
    }

    const bool gridSizeChanged = TRUE; /* This function is called on DLB steps as well */
    if (gridSizeChanged)               /* The need for reallocation is checked for inside, might do a redundant grid size increase check here anyway?... */
    {
        pme->gpu->kernelParams.grid.ewaldFactor = (M_PI * M_PI) / (pme->ewaldcoeff_q * pme->ewaldcoeff_q);

        /* The grid size variants */
        const int3   localGridSize = {pme->nkx, pme->nky, pme->nkz};
        memcpy(&pme->gpu->kernelParams.grid.localGridSize, &localGridSize, sizeof(localGridSize));
        const float3 localGridSizeFP = {(float)localGridSize.x, (float)localGridSize.y, (float)localGridSize.z};
        memcpy(&pme->gpu->kernelParams.grid.localGridSizeFP, &localGridSizeFP, sizeof(localGridSizeFP));
        const int3   localGridSizePadded = {pme->pmegrid_nx, pme->pmegrid_ny, pme->pmegrid_nz};
        memcpy(&pme->gpu->kernelParams.grid.localGridSizePadded, &localGridSizePadded, sizeof(localGridSizePadded));

        pme_gpu_copy_wrap_zones(pme);
        pme_gpu_realloc_and_copy_fract_shifts(pme);
        pme_gpu_realloc_and_copy_bspline_values(pme);
        pme_gpu_realloc_grids(pme);

        if (pme->gpu->bGPUFFT)
        {
            snew(pme->gpu->pfft_setup_gpu, pme->ngrids); //yupinov - memory leaking?
            for (int i = 0; i < pme->ngrids; ++i)
            {
                gmx_parallel_3dfft_init_gpu(&pme->gpu->pfft_setup_gpu[i], (int *)&localGridSize, pme);
            }
        }
    }

    pme_gpu_step_reinit(pme);
}

void pme_gpu_destroy(gmx_pme_t *pme)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    stopGpuProfiler();

    cudaError_t stat;

    /* Free lots of dynamic data */
    pme_gpu_free_energy_virial(pme);
    pme_gpu_free_bspline_values(pme);
    pme_gpu_free_forces(pme);
    pme_gpu_free_coordinates(pme);
    pme_gpu_free_charges(pme);
    pme_gpu_free_spline_data(pme);
    pme_gpu_free_grid_indices(pme);
    pme_gpu_free_fract_shifts(pme);
    pme_gpu_free_grids(pme);

    /* cuFFT cleanup */
    if (pme->gpu->pfft_setup_gpu)
    {
        for (int i = 0; i < pme->ngrids; i++)
        {
            gmx_parallel_3dfft_destroy_gpu(pme->gpu->pfft_setup_gpu[i]);
        }
        sfree(pme->gpu->pfft_setup_gpu);
    }

    /* Free the synchronization events */
    stat = cudaEventDestroy(pme->gpu->syncEnerVirD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncEnerVirH2D");
    stat = cudaEventDestroy(pme->gpu->syncForcesD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncForcesH2D");
    stat = cudaEventDestroy(pme->gpu->syncSpreadGridD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncpreadGridH2D");
    stat = cudaEventDestroy(pme->gpu->syncSolveGridD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncSolveGridH2D");

    /* Free the timing events */
    pme_gpu_destroy_timings(pme);

    /* Destroy the CUDA stream */
    stat = cudaStreamDestroy(pme->gpu->pmeStream);
    CU_RET_ERR(stat, "PME cudaStreamDestroy error");

    /* Finally free the GPU structure itself */
    sfree(pme->gpu);
    pme->gpu = NULL;
}

void pme_gpu_set_io_ranges(const gmx_pme_t *pme, rvec *coordinates, rvec *forces)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    pme->gpu->forcesHost       = reinterpret_cast<float *>(forces);
    pme->gpu->coordinatesHost  = reinterpret_cast<float *>(coordinates);
    /* TODO: should the cudaHostRegister be called for the *Host pointers under some condition/policy? */
}

void pme_gpu_start_step(const gmx_pme_t *pme, const matrix box)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    pme_gpu_copy_coordinates(pme);

    const size_t   boxMemorySize        = sizeof(matrix);
    const gmx_bool haveToUpdateUnitCell = memcmp(pme->gpu->previousBox, box, boxMemorySize);
    /* There could be a pressure coupling check here, but this is more straightforward.
     * This is an exact comparison of float values though.
     */
    if (haveToUpdateUnitCell)
    {
        memcpy(pme->gpu->previousBox, box, boxMemorySize);

        pme->gpu->kernelParams.step.boxVolume = box[XX][XX] * box[YY][YY] * box[ZZ][ZZ];
        assert(pme->gpu->kernelParams.step.boxVolume != 0.0f);

        matrix recipBox;
        gmx::invertBoxMatrix(box, recipBox);
        /* The GPU recipBox is transposed as compared to the CPU recipBox.
         * Spread uses matrix columns (while solve and gather use rows).
         * There is no particular reason for this; it might be further rethought/optimized for better access patterns.
         */
        const float3 newRecipBox[DIM] =
        {
            {recipBox[XX][XX], recipBox[YY][XX], recipBox[ZZ][XX]},
            {             0.0, recipBox[YY][YY], recipBox[ZZ][YY]},
            {             0.0,              0.0, recipBox[ZZ][ZZ]}
        };
        memcpy(pme->gpu->kernelParams.step.recipBox, newRecipBox, boxMemorySize);
    }
}

void pme_gpu_reinit_atoms(const gmx_pme_t *pme, const int nAtoms, float *coefficients)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    pme->gpu->kernelParams.atoms.nAtoms = nAtoms;
    const int      alignment = 8; //yupinov FIXME: this is particlesPerBlock
    pme->gpu->nAtomsPadded = ((nAtoms + alignment - 1) / alignment) * alignment;
    int            nAtomsAlloc   = PME_GPU_USE_PADDING ? pme->gpu->nAtomsPadded : nAtoms;
    const gmx_bool haveToRealloc = (pme->gpu->nAtomsAlloc < nAtomsAlloc); /* This check might be redundant, but is logical */
    pme->gpu->nAtomsAlloc = nAtomsAlloc;

    pme->gpu->coefficientsHost = reinterpret_cast<float *>(coefficients);
    pme_gpu_realloc_and_copy_coefficients(pme); /* could also be checked for haveToRealloc, but the copy always needs to be performed */

    if (haveToRealloc)
    {
        pme_gpu_realloc_coordinates(pme);
        pme_gpu_realloc_forces(pme);
        pme_gpu_realloc_spline_data(pme);
        pme_gpu_realloc_grid_indices(pme);
    }
}

void pme_gpu_init_atoms_once(const gmx_pme_t *pme, const int nAtoms, float *coefficients)
{
    if (pme->gpu->bNeedToUpdateAtoms)
    {
        pme_gpu_reinit_atoms(pme, nAtoms, coefficients);
        pme->gpu->bNeedToUpdateAtoms = FALSE;
    }
}

/*! \brief \internal
 * Waits for the PME GPU output virial/energy copy to the intermediate CPU buffer to finish.
 *
 * \param[in] pme  The PME structure.
 */
void pme_gpu_sync_energy_virial(const gmx_pme_t *pme)
{
    cudaError_t stat = cudaStreamWaitEvent(pme->gpu->pmeStream, pme->gpu->syncEnerVirD2H, 0);
    CU_RET_ERR(stat, "Error while waiting for PME solve");

    for (int j = 0; j < PME_GPU_VIRIAL_AND_ENERGY_COUNT; j++)
    {
        GMX_ASSERT(!isnan(pme->gpu->virialAndEnergyHost[j]), "PME GPU produces incorrect energy/virial.");
    }
}

void pme_gpu_finish_step(const gmx_pme_t *pme, const gmx_bool bCalcF, const gmx_bool bCalcEnerVir)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    cudaError_t stat = cudaStreamSynchronize(pme->gpu->pmeStream); /* Needed for copy back/timing events */
    CU_RET_ERR(stat, "Failed to synchronize the PME GPU stream!");

    if (bCalcF)
    {
        pme_gpu_sync_output_forces(pme);
    }
    if (bCalcEnerVir)
    {
        pme_gpu_sync_energy_virial(pme);
    }

    pme_gpu_update_timings(pme);

    pme_gpu_step_reinit(pme);
}

/* FIXME: this function does not actually seem to be used when it should be, with CPU FFT? */
void pme_gpu_sync_grid(const gmx_pme_t *pme, const gmx_fft_direction dir)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    gmx_bool syncGPUGrid = ((dir == GMX_FFT_REAL_TO_COMPLEX) ? true : pme->gpu->bGPUSolve);
    if (syncGPUGrid)
    {
        cudaError_t stat = cudaStreamWaitEvent(pme->gpu->pmeStream,
                                               (dir == GMX_FFT_REAL_TO_COMPLEX) ? pme->gpu->syncSpreadGridD2H : pme->gpu->syncSolveGridD2H, 0);
        CU_RET_ERR(stat, "Error while waiting for the PME GPU grid to be copied to CPU");
    }
}

// TODO: use gmx_inline for small functions

// wrappers just for the pme.cpp host calls - a PME GPU code that should ideally be in this file as well
// C++11 not supported in CUDA host code by default => the code stays there for now

gmx_bool pme_gpu_performs_gather(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && pme->gpu->bGPUGather;
}

gmx_bool pme_gpu_performs_FFT(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && pme->gpu->bGPUFFT;
}

gmx_bool pme_gpu_performs_wrapping(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && pme->gpu->bGPUSingle;
}

gmx_bool pme_gpu_performs_solve(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && pme->gpu->bGPUSolve;
}

/*! \brief \internal
 * A convenience wrapper for launching either the GPU or CPU FFT.
 *
 * \param[in] pme            The PME structure.
 * \param[in] grid_index     The grid index - would currently always be 0.
 * \param[in] dir            The FFT direction enum.
 * \param[in] wcycle         The wallclock counter.
 */
void gmx_parallel_3dfft_execute_gpu_wrapper(gmx_pme_t              *pme,
                                            const int               grid_index,
                                            enum gmx_fft_direction  dir,
                                            gmx_wallcycle_t         wcycle)
{
    if (pme_gpu_performs_FFT(pme))
    {
        wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_PME);
        pme_gpu_3dfft(pme, dir, grid_index);
        wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_PME);
    }
    else
    {
        wallcycle_start(wcycle, ewcPME_FFT);
        //TODO: suppress warnins on non-OpenMP build?
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
        for (int thread = 0; thread < pme->nthread; thread++)
        {
            gmx_parallel_3dfft_execute(pme->pfft_setup[grid_index], dir, thread, wcycle);
        }
        wallcycle_stop(wcycle, ewcPME_FFT);
    }
}

/* Finally, the actual PME step code.
 * pme_gpu_launch starts the single PME GPU step work.
 * pme_gpu_get_results waits for the PME GPU step work to complete and fetches the output (forces/energy/virial).
 * Together, they are a GPU counterpart to gmx_pme_do, albeit cut down due to unsupported features
 * (see pme_gpu_check_restrictions).
 *
 * There is also a separate gather launch for now...
 */

void pme_gpu_launch(gmx_pme_t      *pme,
                    int             nAtoms,
                    rvec            x[],
                    rvec            f[],
                    real            charges[],
                    matrix          box,
                    gmx_wallcycle_t wcycle,
                    int             flags)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    pmegrids_t          *pmegrid     = NULL;
    real                *grid        = NULL;
    real                *fftgrid;
    t_complex           *cfftgrid;
    int                  thread = 0;
    gmx_bool             bFirst, bDoSplines;
    const gmx_bool       bCalcEnerVir            = flags & GMX_PME_CALC_ENER_VIR;
    const gmx_bool       bBackFFT                = flags & (GMX_PME_CALC_F | GMX_PME_CALC_POT);

    assert(pme->nnodes > 0);
    assert(pme->nnodes == 1 || pme->ndecompdim > 0);

    bFirst = TRUE;

    wallcycle_sub_start(wcycle, ewcsLAUNCH_GPU_PME);
    pme_gpu_init_atoms_once(pme, nAtoms, charges); /* This only does a one-time atom data init at the first MD step.
                                                    * Additional reinits are called when needed after gmx_pme_recv_coeffs_coords.
                                                    */
    pme_gpu_set_io_ranges(pme, x, f);              /* Should this be called every step, or on DD/DLB, or on bCalcEnerVir change? */
    pme_gpu_start_step(pme, box);                  /* This copies the coordinates, and updates the unit cell box (if it changed) */
    wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_PME);

    /* For simplicity, we construct the splines for all particles if
     * more than one PME calculations is needed. Some optimization
     * could be done by keeping track of which atoms have splines
     * constructed, and construct new splines on each pass for atoms
     * that don't yet have them.
     * For GPU this value currently will be false, possibly increasing the divergence in pme_spline.
     */

    bDoSplines = pme->bFEP || (pme->doCoulomb && pme->doLJ);

    const unsigned int grid_index = 0;

    /* Unpack structure */
    pmegrid     = &pme->pmegrid[grid_index];
    fftgrid     = pme->fftgrid[grid_index];
    cfftgrid    = pme->cfftgrid[grid_index];

    grid = pmegrid->grid.grid;



    // pme->gpu->bGPUSingle && pme->gpu->bGPUFFT should be checked somewhere around here for multi-process
    // pme->gpu->bGPUSolve &&= (grid_index < DO_Q);  // no LJ support
    // no bBackFFT, no bCalcF checks

    if (flags & GMX_PME_SPREAD)
    {
        /* Spread the coefficients on a grid */
        wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_PME);
        // TODO rename: consider using the "pme_gpu" prefix here
        pme_gpu_spread(pme, &pme->atc[0], grid_index, &pmegrid->grid, bFirst, TRUE, bDoSplines);
        wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_PME);

        //if (!pme->bUseThreads)
        {
            if (!pme_gpu_performs_wrapping(pme))
            {
                wrap_periodic_pmegrid(pme, grid);
            }

            /* sum contributions to local grid from other nodes */
#if GMX_MPI
            if (pme->nnodes > 1)
            {
                gmx_sum_qgrid_dd(pme, grid, GMX_SUM_GRID_FORWARD);
                where();
            }
#endif
            if (!pme_gpu_performs_FFT(pme))
            {
                copy_pmegrid_to_fftgrid(pme, grid, fftgrid, grid_index);
            }
        }
    }

    try
    {
        if (flags & GMX_PME_SOLVE)
        {
            /* do 3d-fft */
            gmx_parallel_3dfft_execute_gpu_wrapper(pme, grid_index, GMX_FFT_REAL_TO_COMPLEX,
                                                   wcycle);

            /* solve in k-space for our local cells */
            if (pme_gpu_performs_solve(pme))
            {
                wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_PME);
                pme_gpu_solve(pme, cfftgrid, bCalcEnerVir);
                wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_PME);
            }
            else
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
            {
                for (thread = 0; thread < pme->nthread; thread++)
                {
                    solve_pme_yzx(pme, cfftgrid,
                                  box[XX][XX]*box[YY][YY]*box[ZZ][ZZ],
                                  bCalcEnerVir, pme->nthread, thread);
                }
            }
        }

        if (bBackFFT)
        {
            /* do 3d-invfft */
            gmx_parallel_3dfft_execute_gpu_wrapper(pme, grid_index, GMX_FFT_COMPLEX_TO_REAL, wcycle);

            if (!pme_gpu_performs_FFT(pme) || !pme_gpu_performs_gather(pme))
            {
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
                for (thread = 0; thread < pme->nthread; thread++)
                {
                    copy_fftgrid_to_pmegrid(pme, fftgrid, grid, grid_index, pme->nthread, thread);
                }
            }
        }
    } GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;

    if (bBackFFT)
    {
        /* distribute local grid to all nodes */
        if (!pme_gpu_performs_wrapping(pme))
        {
            unwrap_periodic_pmegrid(pme, grid);
        }
    }
}

// this will only copy the forces buffer (with results from listed calculations, etc.) to the GPU (for bClearF == false),
// launch the gather kernel, copy the result back
void pme_gpu_launch_gather(gmx_pme_t                 *pme,
                           gmx_wallcycle_t gmx_unused wcycle,
                           gmx_bool                   bClearForces)
{
    if (!pme_gpu_performs_gather(pme))
    {
        return;
    }

    wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_PME);
    pme_gpu_gather(pme, bClearForces);
    wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_PME);
}

void pme_gpu_get_results(const gmx_pme_t *pme,
                         gmx_wallcycle_t  wcycle,
                         matrix           vir_q,
                         real            *energy_q,
                         int              flags)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    const gmx_bool       bCalcEnerVir            = flags & GMX_PME_CALC_ENER_VIR;
    const gmx_bool       bCalcF                  = flags & GMX_PME_CALC_F;

    wallcycle_sub_start(wcycle, ewcsWAIT_GPU_PME);
    pme_gpu_finish_step(pme, bCalcF, bCalcEnerVir);
    wallcycle_sub_stop(wcycle, ewcsWAIT_GPU_PME);

    if (bCalcEnerVir)
    {
        if (pme->doCoulomb)
        {
            pme_gpu_get_energy_virial(pme, energy_q, vir_q);
            if (debug)
            {
                fprintf(debug, "Electrostatic PME mesh energy [GPU]: %g\n", *energy_q);
            }
        }
        else
        {
            *energy_q = 0;
        }
    }
    /* No bCalcF code since currently forces are copied to the output host buffer with no perturbation. */
}
