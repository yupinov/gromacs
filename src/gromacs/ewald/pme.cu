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

#include "gromacs/math/units.h"
#include "gromacs/utility/smalloc.h"
#include "pme.cuh"
#include "pme-gpu.h"

static gmx_bool debugMemoryPrint = false;

/* A GPU/host memory deallocation routine */
void PMEMemoryFree(const gmx_pme_t *pme, PMEDataID id, MemLocType location)
{
    cudaError_t stat;
    size_t      i = location * PME_ID_END_INVALID + id;
    if (pme->gpu->StoragePointers[i])
    {
        if (debugMemoryPrint)
        {
            printf("free! %p %d %d\n", pme->gpu->StoragePointers[i], id, location);
        }
        if (location == ML_DEVICE)
        {
            stat = cudaFree(pme->gpu->StoragePointers[i]);
            CU_RET_ERR(stat, "PME cudaFree error");
        }
        else
        {
            stat = cudaFreeHost(pme->gpu->StoragePointers[i]);
            CU_RET_ERR(stat, "PME cudaFreeHost error");
        }
        pme->gpu->StoragePointers[i] = NULL;
    }
}

/* \brief
 *
 * A GPU/host memory allocation/fetching routine.
 * If size is 0, it just returns the current pointer.
 */
void *PMEMemoryFetch(const gmx_pme_t *pme, PMEDataID id, size_t size, MemLocType location)
{
    assert(pme->gpu);
    cudaError_t stat = cudaSuccess;
    size_t      i    = location * PME_ID_END_INVALID + id;

    if (debugMemoryPrint && (pme->gpu->StorageSizes[i] > 0) && (size > 0) && (size > pme->gpu->StorageSizes[i]))
    {
        printf("Asked to reallocate %lu into %lu with ID %d\n", pme->gpu->StorageSizes[i], size, id);
    }

    if (pme->gpu->StorageSizes[i] < size)
    {
        PMEMemoryFree(pme, id, location);
        if (size > 0)
        {
            if (debugMemoryPrint)
            {
                printf("Asked to alloc %lu", size);
            }
            if (location == ML_DEVICE)
            {
                stat = cudaMalloc((void **)&pme->gpu->StoragePointers[i], size);
                CU_RET_ERR(stat, "PME cudaMalloc error");
            }
            else
            {
                unsigned int allocFlags = cudaHostAllocDefault;
                /*
                 * allocFlags |= cudaHostAllocWriteCombined;
                 * Could try cudaHostAllocWriteCombined for almost-constant global memory?
                 * (like coordinates/coefficients and thetas/dthetas)
                 */
                stat = cudaHostAlloc((void **)&pme->gpu->StoragePointers[i], size, allocFlags);
                CU_RET_ERR(stat, "PME cudaHostAlloc error");
            }
            pme->gpu->StorageSizes[i] = size;
        }
    }
    return pme->gpu->StoragePointers[i];
}

/* Copies the reciprocal box to the device (used in PME spread/solve/gather)*/
void pme_gpu_copy_recipbox(gmx_pme_t *pme)
{
    const float3 box[3] =
    {
        {pme->recipbox[XX][XX], pme->recipbox[YY][XX], pme->recipbox[ZZ][XX]},
        {                  0.0, pme->recipbox[YY][YY], pme->recipbox[ZZ][YY]},
        {                  0.0,                   0.0, pme->recipbox[ZZ][ZZ]}
    };
    assert(pme->recipbox[XX][XX] != 0.0);
    memcpy(pme->gpu->constants.recipbox, box, sizeof(box));
}

/* Copies the grid sizes for overlapping (used in the current shabby PME wrap/unwrap code) */
void pme_gpu_copy_wrap_zones(gmx_pme_t *pme)
{
    const int nx      = pme->nkx;
    const int ny      = pme->nky;
    const int nz      = pme->nkz;
    const int overlap = pme->pme_order - 1;

    // cell count in 7 parts of overlap
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

    const int2 zoneSizesYZ_h[OVERLAP_ZONES] =
    {
        {     ny,   overlap},
        {overlap,        nz},
        {     ny,        nz},
        {overlap,   overlap},
        {     ny,   overlap},
        {overlap,        nz},
        {overlap,   overlap}
    };

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
    memcpy(pme->gpu->overlap.overlapSizes, zoneSizesYZ_h, sizeof(zoneSizesYZ_h));
    memcpy(pme->gpu->overlap.overlapCellCounts, cellsAccumCount_h, sizeof(cellsAccumCount_h));
}

/* Copies the coordinates to the device (used in PME spread) */
void pme_gpu_copy_coordinates(gmx_pme_t *pme)
{
    const size_t coordinatesSize = DIM * pme->gpu->constants.nAtoms * sizeof(real);
    float3      *coordinates_h   = (float3 *)PMEMemoryFetch(pme, PME_ID_XPTR, coordinatesSize, ML_HOST);
    memcpy(coordinates_h, pme->atc[0].x, coordinatesSize);
    pme->gpu->coordinates = (float3 *)PMEMemoryFetch(pme, PME_ID_XPTR, coordinatesSize, ML_DEVICE);
    cu_copy_H2D_async(pme->gpu->coordinates, coordinates_h, coordinatesSize, pme->gpu->pmeStream);
}

/* The PME GPU reinitialization function that is called both at the end of any MD step and on any DD step */
void pme_gpu_step_reinit(const gmx_pme_t *pme)
{
    const int grid_index = 0;
    pme_gpu_clear_grid(pme, grid_index);
    pme_gpu_clear_energy_virial(pme, grid_index);
}

/* The PME GPU initialization function that is called in the beginning of the run and on any DD step */
void pme_gpu_init(gmx_pme_gpu_t **pmeGPU, gmx_pme_t *pme, const gmx_hw_info_t *hwinfo,
                  const gmx_gpu_opt_t *gpu_opt)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    const int grid_index = 0;

    gmx_bool  firstInit = !*pmeGPU;
    if (firstInit)
    {
        snew(*pmeGPU, 1);
        cudaError_t stat;

        /* GPU selection copied from non-bondeds */
        const int PMEGPURank = pme->nodeid;
        char      gpu_err_str[STRLEN];
        assert(hwinfo);
        assert(hwinfo->gpu_info.gpu_dev);
        assert(gpu_opt->dev_use);

        int   forcedGpuId     = -1;
        char *forcedGpuIdHack = getenv("GMX_PME_GPU_ID");
        if (forcedGpuIdHack)
        {
            forcedGpuId = atoi(forcedGpuIdHack);
            printf("PME rank %d trying to use GPU %d\n", PMEGPURank, forcedGpuId);
            stat = cudaSetDevice(forcedGpuId);
            CU_RET_ERR(stat, "PME failed to set the GPU device ");
        }
        else
        {
            (*pmeGPU)->deviceInfo = &hwinfo->gpu_info.gpu_dev[gpu_opt->dev_use[PMEGPURank]];
            const gmx::MDLogger temp;
            if (!init_gpu(temp, PMEGPURank, gpu_err_str, &hwinfo->gpu_info, gpu_opt))
            {
                gmx_fatal(FARGS, "Could not select GPU %d for PME rank %d\n", (*pmeGPU)->deviceInfo->id, PMEGPURank);
            }
        }

        // fallback instead?
        // first init and either of the hw structures NULL => should also fall back to CPU

        /* Some permanent settings are set here */

        (*pmeGPU)->bGPUSingle = pme_gpu_enabled(pme) && (pme->nnodes == 1);
        /* A convenience variable. */

        (*pmeGPU)->bGPUFFT = (*pmeGPU)->bGPUSingle && !getenv("GMX_PME_GPU_FFTW");
        /* cuFFT is only used for a single rank. */

        (*pmeGPU)->bGPUSolve = true; //(*pmeGPU)->bGPUFFT;
        /* CPU solve with the CPU FFTW is definitely broken at the moment - 20160511 */

        (*pmeGPU)->bGPUGather = true;
        /* CPU gather has got to be broken as well due to different theta/dtheta layout. */

        (*pmeGPU)->bOutOfPlaceFFT = true;
        /* This should give better performance, according to the cuFFT documentation.
         * The performance seems to be the same though.
         * Perhaps the limiting factor is using paddings/overlaps in the grid, which is also frowned upon.
         * PME could also try to pick up nice grid sizes (with factors of 2, 3, 5, 7)
         */

        (*pmeGPU)->bTiming = (getenv("GMX_DISABLE_CUDA_TIMING") == NULL);
        /* This should also check for NB GPU being launched,
         * and NB should check for PME GPU!
         */

        (*pmeGPU)->useTextureObjects = forcedGpuIdHack ? false : ((*pmeGPU)->deviceInfo->prop.major >= 3);
        // if false, texture references are used instead
        //yupinov - have to fix this GPU id selection for good

        size_t pointerStorageSize = ML_END_INVALID * PME_ID_END_INVALID;
        (*pmeGPU)->StorageSizes.assign(pointerStorageSize, 0);
        (*pmeGPU)->StoragePointers.assign(pointerStorageSize, NULL);

        /* Creating a PME CUDA stream */
#if GMX_CUDA_VERSION >= 5050
        int highest_priority;
        int lowest_priority;
        stat = cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
        CU_RET_ERR(stat, "PME cudaDeviceGetStreamPriorityRange failed");
        stat = cudaStreamCreateWithPriority(&(*pmeGPU)->pmeStream,
                                            //cudaStreamNonBlocking,
                                            cudaStreamDefault,
                                            highest_priority);

        CU_RET_ERR(stat, "cudaStreamCreateWithPriority on PME stream failed");
#else
        stat = cudaStreamCreate(&(*pmeGPU)->pmeStream);
        CU_RET_ERR(stat, "PME cudaStreamCreate error");
#endif

        /* Creating synchronization events */
        stat = cudaEventCreateWithFlags(&(*pmeGPU)->syncEnerVirD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncEnerVirH2D failed");
        stat = cudaEventCreateWithFlags(&(*pmeGPU)->syncForcesD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncForcesH2D failed");
        stat = cudaEventCreateWithFlags(&(*pmeGPU)->syncSpreadGridD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncSpreadGridH2D failed");
        stat = cudaEventCreateWithFlags(&(*pmeGPU)->syncSolveGridD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncSolveGridH2D failed");

        if ((pme->gpu)->bTiming)
        {
            pme_gpu_init_timings(pme);
        }

        /* This has a constant size of 6 + 1 floats */
        pme_gpu_alloc_energy_virial(pme, grid_index);
    }

    const bool gridSizeChanged            = true;
    const bool localParticleNumberChanged = firstInit; /* Should be triggered on PME DD as well! */

    if (gridSizeChanged)
    {
        const int3   localGridSize = {pme->nkx, pme->nky, pme->nkz};
        memcpy(&pme->gpu->constants.localGridSize, &localGridSize, sizeof(localGridSize));
        const float3 localGridSizeFP = {(real)localGridSize.x, (real)localGridSize.y, (real)localGridSize.z};
        memcpy(&pme->gpu->constants.localGridSizeFP, &localGridSizeFP, sizeof(localGridSizeFP));
        const int3   localGridSizePadded = {pme->pmegrid_nx, pme->pmegrid_ny, pme->pmegrid_nz};
        memcpy(&pme->gpu->constants.localGridSizePadded, &localGridSizePadded, sizeof(localGridSizePadded));

        pme_gpu_copy_wrap_zones(pme);
        pme_gpu_copy_calcspline_constants(pme);
        pme_gpu_copy_bspline_moduli(pme);
        pme_gpu_alloc_grids(pme, grid_index);

        if ((*pmeGPU)->bGPUFFT)
        {
            snew((*pmeGPU)->pfft_setup_gpu, pme->ngrids);
            for (int i = 0; i < pme->ngrids; ++i)
            {
                gmx_parallel_3dfft_init_gpu(&(*pmeGPU)->pfft_setup_gpu[i], (int *)&localGridSize, pme);
            }
        }
    }

    if (localParticleNumberChanged)
    {
        pme->gpu->constants.nAtoms = pme->atc[0].n;
        pme_gpu_alloc_gather_forces(pme);
    }

    pme_gpu_step_reinit(pme);
}

/* The PME GPU destructor function that is called at the end of the run*/
void pme_gpu_deinit(gmx_pme_t **pme)
{
    if (!pme_gpu_enabled(*pme)) /* Assuming this boolean doesn't change during the run */
    {
        return;
    }

    stopGpuProfiler();

    cudaError_t stat;

    /* These are all the GPU/host pointers allocated through PMEMemoryFetch - grids included. */
    for (unsigned int id = 0; id < PME_ID_END_INVALID; id++)
    {
        for (unsigned int location = 0; location < ML_END_INVALID; location++)
        {
            PMEMemoryFree(*pme, (PMEDataID)id, (MemLocType)location);
        }
    }

    // FFT cleanup
    if ((*pme)->gpu->pfft_setup_gpu)
    {
        for (int i = 0; i < (*pme)->ngrids; i++)
        {
            gmx_parallel_3dfft_destroy_gpu((*pme)->gpu->pfft_setup_gpu[i]);
        }
        sfree((*pme)->gpu->pfft_setup_gpu);
    }

    // destroy sthe ynchronization events
    stat = cudaEventDestroy((*pme)->gpu->syncEnerVirD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncEnerVirH2D");
    stat = cudaEventDestroy((*pme)->gpu->syncForcesD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncForcesH2D");
    stat = cudaEventDestroy((*pme)->gpu->syncSpreadGridD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncpreadGridH2D");
    stat = cudaEventDestroy((*pme)->gpu->syncSolveGridD2H);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncSolveGridH2D");

    // destroy the timing events
    pme_gpu_destroy_timings(*pme);

    // destroy the stream
    stat = cudaStreamDestroy((*pme)->gpu->pmeStream);
    CU_RET_ERR(stat, "PME cudaStreamDestroy error");

    // delete the structure itself
    sfree((*pme)->gpu);
    (*pme)->gpu = NULL;
}

void pme_gpu_set_constants(gmx_pme_t *pme, const matrix box, const real ewaldCoeff)
{
    // this is ran at the beginning of MD step
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    /* Assuming the recipbox is calculated already */
    pme_gpu_copy_recipbox(pme); // could use some boolean checks to know if it should run each time, like pressure coupling?

    pme->gpu->constants.volume = box[XX][XX] * box[YY][YY] * box[ZZ][ZZ];
    assert(pme->gpu->constants.volume != 0.0f);

    pme->gpu->constants.ewaldFactor = (M_PI * M_PI) / (ewaldCoeff * ewaldCoeff);

    pme->gpu->constants.elFactor = ONE_4PI_EPS0 / pme->epsilon_r;
}


void pme_gpu_step_init(gmx_pme_t *pme)
{
    // this is ran at the beginning of MD step
    // should ideally be empty
    //and now there is also setparam call?
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    pme_gpu_copy_coordinates(pme);
}

void pme_gpu_grid_init(const gmx_pme_t *pme, const int gmx_unused grid_index)
{
    // this is ran at the beginning of MD step
    // should ideally be empty
    //and now there is also setparam call?
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    pme_gpu_copy_charges(pme);
}

void pme_gpu_step_end(const gmx_pme_t *pme, const gmx_bool bCalcF, const gmx_bool bCalcEnerVir)
{
    // this is ran at the end of MD step
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    cudaError_t stat = cudaStreamSynchronize(pme->gpu->pmeStream);
    // needed for timings and for copy back events
    CU_RET_ERR(stat, "failed to synchronize the PME GPU stream!");

    if (bCalcF)
    {
        pme_gpu_get_forces(pme);
    }
    if (bCalcEnerVir)
    {
        pme_gpu_get_energy_virial(pme);
    }

    pme_gpu_update_timings(pme);

    pme_gpu_step_reinit(pme);
}

void pme_gpu_copy_charges(const gmx_pme_t *pme)
{
    const size_t coefficientSize = pme->gpu->constants.nAtoms * sizeof(real);
    real        *coefficients_h  = (real *)PMEMemoryFetch(pme, PME_ID_COEFFICIENT, coefficientSize, ML_HOST);
    memcpy(coefficients_h, pme->atc[0].coefficient, coefficientSize); // why not just register host memory?
    pme->gpu->coefficients = (real *)PMEMemoryFetch(pme, PME_ID_COEFFICIENT, coefficientSize, ML_DEVICE);
    cu_copy_H2D_async(pme->gpu->coefficients, coefficients_h, coefficientSize, pme->gpu->pmeStream);
}

void pme_gpu_sync_grid(const gmx_pme_t *pme, gmx_fft_direction dir)
{
    gmx_bool syncGPUGrid = pme_gpu_enabled(pme) && ((dir == GMX_FFT_REAL_TO_COMPLEX) ? true : pme->gpu->bGPUSolve);
    if (syncGPUGrid)
    {
        cudaError_t stat = cudaStreamWaitEvent(pme->gpu->pmeStream,
                                               (dir == GMX_FFT_REAL_TO_COMPLEX) ? pme->gpu->syncSpreadGridD2H : pme->gpu->syncSolveGridD2H, 0);
        CU_RET_ERR(stat, "error while waiting for the GPU grid");
    }
}


gmx_bool pme_gpu_enabled(const gmx_pme_t *pme)
{
    return (pme != NULL) && pme->bGPU;
}

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
