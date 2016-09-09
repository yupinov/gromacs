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
#include "gromacs/math/units.h"
#include "gromacs/utility/smalloc.h"
#include "pme.cuh"
#include "pme-gpu.h"

/*! \brief \internal
 *
 * Allocates the energy and virial memory both on GPU and CPU (7 floats).
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_alloc_energy_virial(const gmx_pme_t *pme)
{
    pme->gpu->energyAndVirialSizeBytes = 7 * sizeof(float); /* 6 virial components + energy */
    cudaError_t stat = cudaMalloc((void **)&pme->gpu->energyAndVirial, pme->gpu->energyAndVirialSizeBytes);
    CU_RET_ERR(stat, "cudaMalloc failed on PME energy and virial");
    pmalloc((void **)&pme->gpu->energyAndVirialHost, pme->gpu->energyAndVirialSizeBytes);
}

/*! \brief \internal
 * Frees the energy and virial memory both on GPU and CPU (7 floats).
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_energy_virial(const gmx_pme_t *pme)
{
    cudaError_t stat = cudaFree(pme->gpu->energyAndVirial);
    CU_RET_ERR(stat, "cudaFree failed on PME energy and virial");
    pme->gpu->energyAndVirial = NULL;
    pfree(pme->gpu->energyAndVirialHost);
    pme->gpu->energyAndVirialHost = NULL;
}

/*! \brief
 *
 * Clears the energy and virial memory on GPU with 0.
 * Should be called at the end of the energy/virial calculation step.
 */
void pme_gpu_clear_energy_virial(const gmx_pme_t *pme)
{
    cudaError_t stat = cudaMemsetAsync(pme->gpu->energyAndVirial, 0, pme->gpu->energyAndVirialSizeBytes, pme->gpu->pmeStream);
    CU_RET_ERR(stat, "PME energies/virial cudaMemsetAsync error");
}

/*! \brief \internal
 * Copies the precalculated reciprocal box to the GPU constants structure.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_copy_recipbox(const gmx_pme_t *pme)
{
    const float3 box[3] =
    {
        {pme->recipbox[XX][XX], pme->recipbox[YY][XX], pme->recipbox[ZZ][XX]},
        {                  0.0, pme->recipbox[YY][YY], pme->recipbox[ZZ][YY]},
        {                  0.0,                   0.0, pme->recipbox[ZZ][ZZ]}
    };
    assert(pme->recipbox[XX][XX] != 0.0f);
    memcpy(pme->gpu->kernelParams.step.recipbox, box, sizeof(box));
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

    const int newSplineValuesSize  = pme->nkx + pme->nky + pme->nkz;
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
    const int3 zoneSizes_h[OVERLAP_ZONES] =
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
    int2 zoneSizesYZ_h[OVERLAP_ZONES];
    for (int i = 0; i < OVERLAP_ZONES; i++)
    {
        zoneSizesYZ_h[i].x = zoneSizes_h[i].y;
        zoneSizesYZ_h[i].y = zoneSizes_h[i].z;
    }
    int cellsAccumCount_h[OVERLAP_ZONES];
    for (int i = 0; i < OVERLAP_ZONES; i++)
    {
        cellsAccumCount_h[i] = zoneSizes_h[i].x * zoneSizes_h[i].y * zoneSizes_h[i].z;
    }
    /* Accumulation */
    for (int i = 1; i < OVERLAP_ZONES; i++)
    {
        cellsAccumCount_h[i] = cellsAccumCount_h[i] + cellsAccumCount_h[i - 1];
    }
    memcpy(pme->gpu->kernelParams.grid.overlapSizes, zoneSizesYZ_h, sizeof(zoneSizesYZ_h));
    memcpy(pme->gpu->kernelParams.grid.overlapCellCounts, cellsAccumCount_h, sizeof(cellsAccumCount_h));
}

/*! \brief
 * Reallocates the GPU buffer for the resulting PME forces.
 *
 *
 */
void pme_gpu_realloc_forces(const gmx_pme_t *pme)
{
    const int newForcesSize = pme->gpu->kernelParams.atoms.nAtoms * DIM;
    assert(pme->gpu->kernelParams.atoms.nAtoms > 0);
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.atoms.forces, NULL, sizeof(float),
                        &pme->gpu->forcesSize, &pme->gpu->forcesSizeAlloc, newForcesSize, pme->gpu->pmeStream, true);
}

void pme_gpu_free_forces(const gmx_pme_t *pme)
{
    cu_free_buffered(pme->gpu->kernelParams.atoms.forces, &pme->gpu->forcesSize, &pme->gpu->forcesSizeAlloc);
}

/*! \brief
 * Reallocates the buffer on the GPU and copies the coordinates from the CPU buffer (pme->gpu->coordinatesHost).
 *
 * \param[in] pme            The PME structure.
 *
 * Needs to be called every MD step. The coordinates are then used in the spline calculation.
 * Should probably be split into realloc (on DD) and copy (every step) parts...
 */
void pme_gpu_realloc_and_copy_coordinates(const gmx_pme_t *pme)
{
    assert(pme->gpu->kernelParams.atoms.nAtoms > 0);
    assert(pme->gpu->coordinatesHost);
    const size_t newCoordinatesSize = pme->gpu->kernelParams.atoms.nAtoms * DIM;
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.atoms.coordinates, pme->gpu->coordinatesHost, sizeof(float),
                        &pme->gpu->coordinatesSize, &pme->gpu->coordinatesSizeAlloc, newCoordinatesSize, pme->gpu->pmeStream, true);
}

/*! \brief
 * Frees the coordinates on the GPU.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_coordinates(const gmx_pme_t *pme)
{
    cu_free_buffered((void **)&pme->gpu->kernelParams.atoms.coordinates, &pme->gpu->coordinatesSize, &pme->gpu->coordinatesSizeAlloc);
}

/*! \brief
 * Reallocates the buffer on the GPU and copies the charges (sometimes also called coefficients) from the CPU buffer (pme->gpu->coefficientsHost).
 *
 * \param[in] pme            The PME structure.
 *
 * Does not need to be done every MD step, only whenever the local charges change.
 * (So, in the beginning of the run, or on DD step).
 */
void pme_gpu_realloc_and_copy_charges(const gmx_pme_t *pme)
{
    assert(pme->gpu->kernelParams.atoms.nAtoms > 0);
    assert(pme->gpu->coefficientsHost);
    const size_t newCoefficientSize = pme->gpu->kernelParams.atoms.nAtoms;
    cu_realloc_buffered((void **)&pme->gpu->kernelParams.atoms.coefficients, pme->gpu->coefficientsHost, sizeof(float),
                        &pme->gpu->coefficientsSize, &pme->gpu->coefficientsSizeAlloc, newCoefficientSize, pme->gpu->pmeStream, true);
}

/*! \brief
 * Frees the charges on the GPU.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_charges(const gmx_pme_t *pme)
{
    cu_free_buffered((void **)&pme->gpu->kernelParams.atoms.coefficients, &pme->gpu->coefficientsSize, &pme->gpu->coefficientsSizeAlloc);
}

/*! \brief
 * Reallocates the buffers on the GPU for the particle spline data.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_realloc_spline_data(const gmx_pme_t *pme)
{
    const int    order     = pme->pme_order;
    const int    alignment = PME_SPREADGATHER_PARTICLES_PER_WARP;
    /* Probably needs to be particlesPerBlock for full padding */
    const size_t nAtomsPadded      = ((pme->gpu->kernelParams.atoms.nAtoms + alignment - 1) / alignment) * alignment;
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

/*! \brief
 * Frees the buffers on the GPU for the particle spline data.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_free_spline_data(const gmx_pme_t *pme)
{
    /* Two arrays of the same size */
    int currentSizeTemp      = pme->gpu->splineDataSize;
    int currentSizeTempAlloc = pme->gpu->splineDataSizeAlloc;
    cu_free_buffered((void **)&pme->gpu->kernelParams.atoms.theta, &currentSizeTemp, &currentSizeTempAlloc);
    cu_free_buffered((void **)&pme->gpu->kernelParams.atoms.dtheta, &pme->gpu->splineDataSize, &pme->gpu->splineDataSizeAlloc);
}

/*! \brief \internal
 * Reallocates the buffer on the GPU for the particle gridline indices.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_realloc_grid_indices(const gmx_pme_t *pme)
{
    const size_t newIndicesSize = DIM * pme->gpu->kernelParams.atoms.nAtoms;
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
    cu_free_buffered((void **)&pme->gpu->kernelParams.atoms.gridlineIndices, &pme->gpu->gridlineIndicesSize, &pme->gpu->gridlineIndicesSizeAlloc);
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
        int tempGridSize      = pme->gpu->gridSize;
        int tempGridSizeAlloc = pme->gpu->gridSizeAlloc;
        cu_free_buffered((void **)&pme->gpu->kernelParams.grid.fourierGrid, &tempGridSize, &tempGridSizeAlloc);
    }
    cu_free_buffered((void **)&pme->gpu->kernelParams.grid.realGrid, &pme->gpu->gridSize, &pme->gpu->gridSizeAlloc);
}

void pme_gpu_clear_grids(const gmx_pme_t *pme)
{
    cudaStream_t s = pme->gpu->pmeStream;

    cudaError_t  stat = cudaMemsetAsync(pme->gpu->kernelParams.grid.realGrid, 0, pme->gpu->gridSize * sizeof(float), s);
    /* Should the complex grid be cleared in some weird case? */
    CU_RET_ERR(stat, "cudaMemsetAsync on the PME grid error");
}

/*! \brief
 * The PME GPU reinitialization function that is called both at the end of any MD step and on any load balancing step.
 *
 * \param[in] pme            The PME structure.
 */
void pme_gpu_step_reinit(const gmx_pme_t *pme)
{
    pme_gpu_clear_grids(pme);
    pme_gpu_clear_energy_virial(pme);
}

/*! \brief
 * The PME GPU initialization function that is called in the beginning of the run and on any load balancing step.
 *
 * \param[in] pme            The PME structure.
 * ......
 */
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

        pme->gpu->bGPUSolve = true;
        /* pme->gpu->bGPUFFT - CPU solve with the CPU FFTW is definitely broken at the moment - 20160511 */

        pme->gpu->bGPUGather = true;
        /* CPU gather has got to be broken as well due to different theta/dtheta layout. */

        pme->gpu->bOutOfPlaceFFT = true;
        /* This should give better performance, according to the cuFFT documentation.
         * The performance seems to be the same though.
         * Perhaps the limiting factor is using paddings/overlaps in the grid, which is also frowned upon.
         * PME could also try to pick up nice grid sizes (with factors of 2, 3, 5, 7)
         */

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

        GMX_RELEASE_ASSERT(pme->epsilon_r != 0.0f, "PME GPU: erroneous electostatic factor");
        pme->gpu->kernelParams.constants.elFactor = ONE_4PI_EPS0 / pme->epsilon_r;
    }

    const bool gridSizeChanged = true; /* This function is called on DLB steps as well */
    if (gridSizeChanged)               /* The need for reallocation is checked for inside, might do a redundant grid size increase check here anyway?... */
    {
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

void pme_gpu_deinit(gmx_pme_t *pme)
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

void pme_gpu_set_constants(const gmx_pme_t *pme, const matrix box, const float ewaldCoeff)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    /* Assuming the recipbox is calculated already */
    pme_gpu_copy_recipbox(pme); // could use some boolean checks to know if it should run each time, like pressure coupling?
    // actually, just compare the memory

    pme->gpu->kernelParams.step.boxVolume = box[XX][XX] * box[YY][YY] * box[ZZ][ZZ];
    assert(pme->gpu->kernelParams.step.boxVolume != 0.0f);

    pme->gpu->kernelParams.grid.ewaldFactor = (M_PI * M_PI) / (ewaldCoeff * ewaldCoeff);
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

void pme_gpu_step_init(const gmx_pme_t *pme)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    pme_gpu_realloc_and_copy_coordinates(pme);
}

void pme_gpu_reinit_atoms(const gmx_pme_t *pme, const int nAtoms, float *coefficients)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    const gmx_bool haveToRealloc = (pme->gpu->kernelParams.atoms.nAtoms < nAtoms);
    pme->gpu->kernelParams.atoms.nAtoms = nAtoms;

    pme->gpu->coefficientsHost = reinterpret_cast<float *>(coefficients);
    pme_gpu_realloc_and_copy_charges(pme);

    if (haveToRealloc) /* This check might be redundant, but is logical */
    {
        pme_gpu_realloc_forces(pme);
        pme_gpu_realloc_spline_data(pme);
        pme_gpu_realloc_grid_indices(pme);
    }
}

void pme_gpu_step_end(const gmx_pme_t *pme, const gmx_bool bCalcF, const gmx_bool bCalcEnerVir)
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
