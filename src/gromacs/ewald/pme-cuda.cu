#include <assert.h>
#include <stdio.h>
#include <vector>

#include "gromacs/gpu_utils/cudautils.cuh"

#include "gromacs/utility/smalloc.h"

// for GPU init
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/utility/logger.h"

#include "pme-cuda.cuh"
#include "pme-gpu.h"

void pme_gpu_step_reinit(gmx_pme_t *pme)
{
    // this is ran at the end of MD step + at the DD init
    const int grid_index = 0; //!
    pme_gpu_clear_grid(pme, grid_index);
    pme_gpu_clear_energy_virial(pme, grid_index);
}

void pme_gpu_init(gmx_pme_gpu_t **pmeGPU, gmx_pme_t *pme, const gmx_hw_info_t *hwinfo,
                  const gmx_gpu_opt_t *gpu_opt)
{
    // this is ran in the beginning/on DD
    if (!pme->bGPU)
        return;

    const int grid_index = 0;

    gmx_bool firstInit = !*pmeGPU;
    if (firstInit)
    {
        // this is only ran once
        snew(*pmeGPU, 1);
        cudaError_t stat;

        // GPU selection copied from non-bondeds
        const int PMEGPURank = pme->nodeid;
        char gpu_err_str[STRLEN];
        assert(hwinfo);
        assert(hwinfo->gpu_info.gpu_dev);
        assert(gpu_opt->dev_use);

        int forcedGpuId = -1;
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
                gmx_fatal(FARGS, "Could not select GPU %d for PME rank %d\n", (*pmeGPU)->deviceInfo->id, PMEGPURank);
        }

        // fallback instead?
        // first init and either of the hw structures NULL => should also fall back to CPU

        // permanent settings

        (*pmeGPU)->bGPUSingle = pme->bGPU && (pme->nnodes == 1);
        // a convenience variable

        (*pmeGPU)->bGPUFFT = (*pmeGPU)->bGPUSingle && !getenv("GMX_PME_GPU_FFTW");
        // currently cuFFT is only used for a single rank

        (*pmeGPU)->bGPUSolve = true; //(*pmeGPU)->bGPUFFT;
        // solve is done between the 2 FFTs - not worth it to copy
        // CPU solve with the CPU FFTW is definitely broken at the moment - 20160511

        (*pmeGPU)->bGPUGather = true;
        // CPU gather has got to be broken - at least fix the spline parameters layout at the end of spread_on_grid_gpu

        (*pmeGPU)->bOutOfPlaceFFT = true;
        // this should give better performance, according to the cuFFT documentation
        // performance seems to be the same though
        // perhaps the limiting factor is using paddings/overlaps in the grid, which is also frowned upon
        // PME should also try to pick up nice grid sizes (with factors of 2, 3, 5, 7)

        (*pmeGPU)->bTiming = (getenv("GMX_DISABLE_CUDA_TIMING") == NULL);
        // this should also check for PP GPU being launched
        // just like NB should check for PME GPU

        (*pmeGPU)->useTextureObjects = forcedGpuIdHack ? false : ((*pmeGPU)->deviceInfo->prop.major >= 3);
        // if false, texture references are used instead
        //yupinov - have to fix this GPU id selection for good

        // internal storage
        size_t pointerStorageSize = ML_END_INVALID * PME_ID_END_INVALID;
        (*pmeGPU)->StorageSizes.assign(pointerStorageSize, 0);
        (*pmeGPU)->StoragePointers.assign(pointerStorageSize, NULL);

        // creating a PME stream
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
        // creating synchronization events
        stat = cudaEventCreateWithFlags(&(*pmeGPU)->syncEnerVirD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncEnerVirH2D failed");
        stat = cudaEventCreateWithFlags(&(*pmeGPU)->syncForcesD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncForcesH2D failed");
        stat = cudaEventCreateWithFlags(&(*pmeGPU)->syncSpreadGridD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncSpreadGridH2D failed");
        stat = cudaEventCreateWithFlags(&(*pmeGPU)->syncSolveGridD2H, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncSolveGridH2D failed");


        if ((pme->gpu)->bTiming)
            pme_gpu_init_timings(pme);

        pme_gpu_alloc_energy_virial(pme, grid_index);
    }

    const bool gridSizeChanged = true;
    const bool localParticleNumberChanged = firstInit; // should be checked for PME DD as well

    if (gridSizeChanged)
    {
        pme_gpu_copy_wrap_zones(pme);
        pme_gpu_copy_calcspline_constants(pme);
        pme_gpu_copy_bspline_moduli(pme);
        pme_gpu_alloc_grids(pme, grid_index);

        if ((*pmeGPU)->bGPUFFT)
        {
            ivec ndata;
            ndata[0] = pme->nkx;
            ndata[1] = pme->nky;
            ndata[2] = pme->nkz;
            snew((*pmeGPU)->pfft_setup_gpu, pme->ngrids);
            for (int i = 0; i < pme->ngrids; ++i)
            {
                gmx_parallel_3dfft_init_gpu(&(*pmeGPU)->pfft_setup_gpu[i], ndata, pme);
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

void pme_gpu_deinit(//gmx_pme_gpu_t **pmeGPU,
                    gmx_pme_t **pme)
{
    // this is ran at the end of run

    if (!(*pme)->bGPU) // we're assuming this boolean doesn't change during the run
        return;

    stopGpuProfiler();

    cudaError_t stat;

    // these are all the GPU/host pointers allocated through PMEMemoryFetch - grids included
    // it's a temporary cleanup solution
    for (unsigned int id = 0; id < PME_ID_END_INVALID; id++)
        for (unsigned int location = 0; location < ML_END_INVALID; location++)
        {
            PMEMemoryFree(*pme, (PMEDataID)id, (MemLocType)location);
        }

    // FFT cleanup
    if ((*pme)->gpu->pfft_setup_gpu)
    {
        for (int i = 0; i < (*pme)->ngrids; i++)
            gmx_parallel_3dfft_destroy_gpu((*pme)->gpu->pfft_setup_gpu[i]);
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

void pme_gpu_step_init(gmx_pme_t *pme)
{
    // this is ran at the beginning of MD step
    // should ideally be empty
    if (!pme->bGPU)
        return;

    pme_gpu_copy_recipbox(pme); // could use some boolean checks, like pressure coupling?

    pme_gpu_copy_coordinates(pme);
}

void pme_gpu_step_end(gmx_pme_t *pme, const gmx_bool bCalcF, const gmx_bool bCalcEnerVir)
{
    // this is ran at the end of MD step
    if (!pme->bGPU)
        return;

    cudaError_t stat = cudaStreamSynchronize(pme->gpu->pmeStream);
    // needed for timings and for copy back events
    CU_RET_ERR(stat, "failed to synchronize the PME GPU stream!");

    if (bCalcF)
        pme_gpu_get_forces(pme);
    if (bCalcEnerVir)
        pme_gpu_get_energy_virial(pme);

    pme_gpu_update_timings(pme);

    pme_gpu_step_reinit(pme);
}

#if PME_EXTERN_CMEM
__constant__ __device__ int2 OVERLAP_SIZES[OVERLAP_ZONES];
__constant__ __device__ int OVERLAP_CELLS_COUNTS[OVERLAP_ZONES];
__constant__ __device__ float3 RECIPBOX[3];
#endif

void pme_gpu_copy_recipbox(gmx_pme_t *pme)
{
    const float3 box[3] =
    {
        {pme->recipbox[XX][XX], pme->recipbox[YY][XX], pme->recipbox[ZZ][XX]},
        {                  0.0, pme->recipbox[YY][YY], pme->recipbox[ZZ][YY]},
        {                  0.0,                   0.0, pme->recipbox[ZZ][ZZ]}
    };
    assert(pme->recipbox[XX][XX] != 0.0);
#if PME_EXTERN_CMEM
    PMECopyConstant(RECIPBOX, box, sizeof(box), s);
#else
    memcpy(pme->gpu->recipbox.box, box, sizeof(box));
#endif
}

void pme_gpu_copy_coordinates(gmx_pme_t *pme)
{
    // coordinates
    const size_t coordinatesSize = DIM * pme->gpu->constants.nAtoms * sizeof(real);
    float3 *coordinates_h = (float3 *)PMEMemoryFetch(pme, PME_ID_XPTR, coordinatesSize, ML_HOST);
    memcpy(coordinates_h, pme->atc[0].x, coordinatesSize);
    pme->gpu->coordinates = (float3 *)PMEMemoryFetch(pme, PME_ID_XPTR, coordinatesSize, ML_DEVICE);
    cu_copy_H2D_async(pme->gpu->coordinates, coordinates_h, coordinatesSize, pme->gpu->pmeStream);
    /*
    float4 *xptr_h = (float4 *)PMEMemoryFetch(pme, PME_ID_XPTR, 4 * n_blocked * sizeof(real), ML_HOST);
    memset(xptr_h, 0, 4 * n_blocked * sizeof(real));
    for (int i = 0; i < pme->gpu->constants.nAtoms; i++)
    {
       memcpy(xptr_h + i, atc->x + i, sizeof(rvec));
    }
    xptr_d = (float4 *)PMEMemoryFetch(pme, PME_ID_XPTR, 4 * n_blocked * sizeof(real), ML_DEVICE);
    PMECopy(pme->gpu->coordinates, xptr_h, 4 * n_blocked * sizeof(real), ML_DEVICE, pme->gpu->pmeStream);
    */
}

void pme_gpu_copy_charges(gmx_pme_t *pme)
{
    // coefficients - can be different for PME/LJ?
    const size_t coefficientSize = pme->gpu->constants.nAtoms * sizeof(real);
    real *coefficients_h = (real *)PMEMemoryFetch(pme, PME_ID_COEFFICIENT, coefficientSize, ML_HOST);
    memcpy(coefficients_h, pme->atc[0].coefficient, coefficientSize); // why not just register host memory?
    pme->gpu->coefficients = (real *)PMEMemoryFetch(pme, PME_ID_COEFFICIENT, coefficientSize, ML_DEVICE);
    cu_copy_H2D_async(pme->gpu->coefficients, coefficients_h, coefficientSize, pme->gpu->pmeStream);
}

void pme_gpu_sync_grid(gmx_pme_t *pme, gmx_fft_direction dir)
{
    gmx_bool syncGPUGrid = pme->bGPU && ((dir == GMX_FFT_REAL_TO_COMPLEX) ? true: pme->gpu->bGPUSolve);
    if (syncGPUGrid)
    {
        cudaError_t stat = cudaStreamWaitEvent(pme->gpu->pmeStream,
            (dir == GMX_FFT_REAL_TO_COMPLEX) ? pme->gpu->syncSpreadGridD2H : pme->gpu->syncSolveGridD2H, 0);
        CU_RET_ERR(stat, "error while waiting for the GPU grid");
    }
}

void pme_gpu_copy_wrap_zones(gmx_pme_t *pme)
{
    const int nx = pme->nkx;
    const int ny = pme->nky;
    const int nz = pme->nkz;
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
        cellsAccumCount_h[i] = zoneSizes_h[i].x * zoneSizes_h[i].y * zoneSizes_h[i].z;
    // accumulate
    for (int i = 1; i < OVERLAP_ZONES; i++)
    {
        cellsAccumCount_h[i] = cellsAccumCount_h[i] + cellsAccumCount_h[i - 1];
    }
#if PME_EXTERN_CMEM
    PMECopyConstant(OVERLAP_SIZES, zoneSizesYZ_h, sizeof(zoneSizesYZ_h), s);
    PMECopyConstant(OVERLAP_CELLS_COUNTS, cellsAccumCount_h, sizeof(cellsAccumCount_h), s);
#else
    memcpy(pme->gpu->overlap.overlapSizes, zoneSizesYZ_h, sizeof(zoneSizesYZ_h));
    memcpy(pme->gpu->overlap.overlapCellCounts, cellsAccumCount_h, sizeof(cellsAccumCount_h));
#endif
}

// wrappers just for the pme.cpp host calls - a PME GPU code that should ideally be in this file as well
// C++11 not supported in CUDA host code by default => the code stays there for now

gmx_bool pme_gpu_performs_gather(gmx_pme_t *pme)
{
    return pme && pme->bGPU && pme->gpu->bGPUGather;
}

gmx_bool pme_gpu_performs_FFT(gmx_pme_t *pme)
{
    return pme && pme->bGPU && pme->gpu->bGPUFFT;
}

gmx_bool pme_gpu_performs_wrapping(gmx_pme_t *pme)
{
    return pme && pme->bGPU && pme->gpu->bGPUSingle;
}

gmx_bool pme_gpu_performs_solve(gmx_pme_t *pme)
{
    return pme && pme->bGPU && pme->gpu->bGPUSolve;
}

// some memory routine wrappers below

static gmx_bool debugMemoryPrint = false;

void PMEMemoryFree(gmx_pme_t *pme, PMEDataID id, MemLocType location)
{
    cudaError_t stat;
    size_t i = location * PME_ID_END_INVALID + id;
    if (pme->gpu->StoragePointers[i])
    {
        if (debugMemoryPrint)
            printf("free! %p %d %d\n", pme->gpu->StoragePointers[i], id, location);
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

void *PMEMemoryFetch(gmx_pme_t *pme, PMEDataID id, size_t size, MemLocType location)
{
    // size == 0 => just return a current pointer

    assert(pme->gpu);
    cudaError_t stat = cudaSuccess;
    size_t i = location * PME_ID_END_INVALID + id;

    if ((pme->gpu->StorageSizes[i] > 0) && (size > 0) && (size > pme->gpu->StorageSizes[i]))
        printf("asked to realloc %lu into %lu with ID %d\n", pme->gpu->StorageSizes[i], size, id);

    if (pme->gpu->StorageSizes[i] < size) // dealloc
    {
        PMEMemoryFree(pme, id, location);
        if (size > 0)
        {
            if (debugMemoryPrint)
                printf("asked to alloc %lu", size);
            size = size * 1.02; // slight overalloc for no apparent reason
            if (debugMemoryPrint)
                printf(", actually allocating %lu\n", size);
            if (location == ML_DEVICE)
            {
                stat = cudaMalloc((void **)&pme->gpu->StoragePointers[i], size);
                CU_RET_ERR(stat, "PME cudaMalloc error");
            }
            else
            {
                unsigned int allocFlags = cudaHostAllocDefault;
                //allocFlags |= cudaHostAllocWriteCombined;
                // try cudaHostAllocWriteCombined for almost-constant global memory?
                // (like coordinates/coefficients and thetas/dthetas)
                // could be helpful for spread being bottlenecked by the memory throughput on Kepler
                stat = cudaHostAlloc((void **)&pme->gpu->StoragePointers[i], size, allocFlags);
                CU_RET_ERR(stat, "PME cudaHostAlloc error");
            }
            pme->gpu->StorageSizes[i] = size;
        }
    }
    return pme->gpu->StoragePointers[i];
}

void PMECopyConstant(const void *dest, void const *src, size_t size, cudaStream_t s)
{
    assert(s != 0);
    cudaError_t stat = cudaMemcpyToSymbolAsync(dest, src, size, 0, cudaMemcpyHostToDevice, s);
    CU_RET_ERR(stat, "PME cudaMemcpyToSymbolAsync");
}

