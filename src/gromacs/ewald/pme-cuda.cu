#include <assert.h>
#include <stdio.h>
#include <vector>

#include "gromacs/gpu_utils/cudautils.cuh"

#include "gromacs/utility/smalloc.h"

// for GPU init
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/hardware/hw_info.h"

#include "pme-cuda.cuh"
#include "pme-gpu.h"

static std::vector<size_t> PMEStorageSizes(ML_END_INVALID * PME_ID_END_INVALID);
static std::vector<void *> PMEStoragePointers(ML_END_INVALID * PME_ID_END_INVALID);


void pme_gpu_update_flags(
        gmx_pme_gpu_t *pmeGPU,
        gmx_bool keepGPUDataBetweenSpreadAndR2C,
        gmx_bool keepGPUDataBetweenR2CAndSolve,
        gmx_bool keepGPUDataBetweenSolveAndC2R,
        gmx_bool keepGPUDataBetweenC2RAndGather
        )
{
    pmeGPU->keepGPUDataBetweenSpreadAndR2C = keepGPUDataBetweenSpreadAndR2C;
    pmeGPU->keepGPUDataBetweenR2CAndSolve = keepGPUDataBetweenR2CAndSolve;
    pmeGPU->keepGPUDataBetweenSolveAndC2R = keepGPUDataBetweenSolveAndC2R;
    pmeGPU->keepGPUDataBetweenC2RAndGather = keepGPUDataBetweenC2RAndGather;
}

void pme_gpu_step_reinit(gmx_pme_t *pme);

void pme_gpu_init(gmx_pme_gpu_t **pmeGPU, gmx_pme_t *pme, const gmx_hw_info_t *hwinfo, const gmx_gpu_opt_t *gpu_opt)
{
    // this is ran in the beginning/on DD
    if (!pme->bGPU) //yupinov fix this
        return;

    gmx_bool firstInit = !*pmeGPU;
    if (firstInit) // first init
    {
        *pmeGPU = new gmx_pme_gpu_t;
        cudaError_t stat;

        PMEStorageSizes.assign(PMEStorageSizes.size(), 0);
        PMEStoragePointers.assign(PMEStoragePointers.size(), NULL);

        // crude GPU selection copied from non-bondeds
        const int PMEGPURank = 0; //yupinov !
        FILE *fp = NULL; //yupinov pass this
        char gpu_err_str[STRLEN];
        assert(hwinfo->gpu_info.gpu_dev);
        assert(gpu_opt->dev_use);
        (*pmeGPU)->deviceInfo = &hwinfo->gpu_info.gpu_dev[gpu_opt->dev_use[PMEGPURank]];
        if (!init_gpu(fp, PMEGPURank, gpu_err_str, &hwinfo->gpu_info, gpu_opt))
            gmx_fatal(FARGS, "could not select GPU %d for PME rank %d\n", (*pmeGPU)->deviceInfo->id, PMEGPURank);
        // fallback instead?
        //first init and either of the hw structures NULL => also fall back to CPU

        (*pmeGPU)->useTextureObjects = ((*pmeGPU)->deviceInfo->prop.major >= 3);


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
        stat = cudaEventCreateWithFlags(&(*pmeGPU)->syncEnerVirH2D, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncEnerVirH2D failed");
        stat = cudaEventCreateWithFlags(&(*pmeGPU)->syncForcesH2D, cudaEventDisableTiming);
        CU_RET_ERR(stat, "cudaEventCreate on syncForcesH2D failed");

        pme_gpu_update_flags(*pmeGPU, false, false, false, false);
    }

    // all these functions should only be called when the grid size changes (e.g. DD)
    const int grid_index = 0;
    pme_gpu_copy_wrap_zones(pme);
    pme_gpu_copy_calcspline_constants(pme);
    pme_gpu_copy_bspline_moduli(pme);
    pme_gpu_alloc_gather_forces(pme);
    pme_gpu_alloc_grid(pme, grid_index);
    pme_gpu_alloc_energy_virial(pme, grid_index);

    if (pme->bGPUFFT) //copied from gmx_pme_init
    {
        ivec ndata;
        ndata[0]    = pme->nkx;
        ndata[1]    = pme->nky;
        ndata[2]    = pme->nkz;
        const gmx_bool bReproducible = false;
        for (int i = 0; i < pme->ngrids; ++i)
        {
            /*
            if ((i <  DO_Q && EEL_PME(ir->coulombtype) && (i == 0 ||
                                                           bFreeEnergy_q)) ||
                (i >= DO_Q && EVDW_PME(ir->vdwtype) && (i == 2 ||
                                                        bFreeEnergy_lj ||
                                                        ir->ljpme_combination_rule == eljpmeLB)))
            */
            if (pme->pfft_setup[i])  //yupinov does not do proper separate init
            {
                 gmx_parallel_3dfft_init_gpu(&pme->pfft_setup_gpu[i], ndata,
                                                 &pme->fftgrid[i], &pme->cfftgrid[i],
                                                 pme->mpi_comm_d,
                                                 bReproducible, pme);

            }
        }
    }

    pme_gpu_step_reinit(pme);

    if (debug)
        fprintf(debug, "PME GPU %s\n", firstInit ? "init" : "reinit");
}

void pme_gpu_deinit(//gmx_pme_gpu_t **pmeGPU,
                    gmx_pme_t **pme)
{
    // this is ran at the end of run

    if (!(*pme)->bGPU) //yupinov - could this boolean change during the run?
        return;

    cudaError_t stat;

    // these are all the GPU/host pointers allocated through PMEMemoryFetch - grids included
    // it's a temporary solution
    for (unsigned int id = 0; id < PME_ID_END_INVALID; id++)
        for (unsigned int location = 0; location < ML_END_INVALID; location++)
        {
            PMEMemoryFree((PMEDataID)id, (MemLocType)location);
        }

    // FFT
    for (int i = 0; i < (*pme)->ngrids; i++)
        gmx_parallel_3dfft_destroy_gpu((*pme)->pfft_setup_gpu[i]);
    sfree((*pme)->pfft_setup_gpu);

    // destroy synchronization events
    stat = cudaEventDestroy((*pme)->gpu->syncEnerVirH2D);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncEnerVirH2D");
    stat = cudaEventDestroy((*pme)->gpu->syncForcesH2D);
    CU_RET_ERR(stat, "cudaEventDestroy failed on syncForcesH2D");

    // destroy the stream
    stat = cudaStreamDestroy((*pme)->gpu->pmeStream);
    CU_RET_ERR(stat, "PME cudaStreamDestroy error");

    // delete the structure itself
    delete ((*pme)->gpu);
    (*pme)->gpu = NULL;
}


void pme_gpu_step_init(gmx_pme_t *pme)
{
    // this is ran at the beginning of MD step
    // should ideally be empty
    if (!pme->bGPU)
        return;

    pme_gpu_copy_recipbox(pme); //yupinov test changing box

    pme_gpu_copy_coordinates(pme);
}

void pme_gpu_step_end(gmx_pme_t *pme, const gmx_bool bCalcF, const gmx_bool bCalcEnerVir)
{
    // this is ran at the end of MD step
    if (!pme->bGPU)
        return;

    cudaError_t stat = cudaStreamSynchronize(pme->gpu->pmeStream); //neede for timings and for copy back events
    CU_RET_ERR(stat, "failed to synchronize the PME GPU stream!");

    if (bCalcF)
        pme_gpu_get_forces(pme);
    if (bCalcEnerVir)
        pme_gpu_get_energy_virial(pme);

    pme_gpu_timing_calculate(pme);

    pme_gpu_step_reinit(pme);
}

void pme_gpu_step_reinit(gmx_pme_t *pme)
{
    // this is ran at the end of MD step + at the DD init
    const int grid_index = 0; //!
    pme_gpu_clear_grid(pme, grid_index);
    pme_gpu_clear_energy_virial(pme, grid_index);
}


#if PME_EXTERN_CMEM
__constant__ __device__ int2 OVERLAP_SIZES[OVERLAP_ZONES];
__constant__ __device__ int OVERLAP_CELLS_COUNTS[OVERLAP_ZONES];
__constant__ __device__ float3 RECIPBOX[3];
#endif

//yupinov stuff more data into constants, like ewaldcoef, etc?

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
    const int n = pme->atc[0].n;

    // coordinates
    const size_t coordinatesSize = DIM * n * sizeof(real);
    float3 *coordinates_h = (float3 *)PMEMemoryFetch(PME_ID_XPTR, coordinatesSize, ML_HOST);
    memcpy(coordinates_h, pme->atc[0].x, coordinatesSize);
    pme->gpu->coordinates = (float3 *)PMEMemoryFetch(PME_ID_XPTR, coordinatesSize, ML_DEVICE);
    cu_copy_H2D_async(pme->gpu->coordinates, coordinates_h, coordinatesSize, pme->gpu->pmeStream);
    /*
    float4 *xptr_h = (float4 *)PMEMemoryFetch(PME_ID_XPTR, 4 * n_blocked * sizeof(real), ML_HOST);
    memset(xptr_h, 0, 4 * n_blocked * sizeof(real));
    for (int i = 0; i < n; i++)
    {
       memcpy(xptr_h + i, atc->x + i, sizeof(rvec));
    }
    xptr_d = (float4 *)PMEMemoryFetch(PME_ID_XPTR, 4 * n_blocked * sizeof(real), ML_DEVICE);
    PMECopy(pme->gpu->coordinates, xptr_h, 4 * n_blocked * sizeof(real), ML_DEVICE, pme->gpu->pmeStream);
    */
}

void pme_gpu_copy_charges(gmx_pme_t *pme)
{
    const int n = pme->atc[0].n;
    // coefficients - can be different for PME/LJ?
    const size_t coefficientSize = n * sizeof(real);
    real *coefficients_h = (real *)PMEMemoryFetch(PME_ID_COEFFICIENT, coefficientSize, ML_HOST);
    memcpy(coefficients_h, pme->atc[0].coefficient, coefficientSize); // why not just register host memory?
    pme->gpu->coefficients = (real *)PMEMemoryFetch(PME_ID_COEFFICIENT, coefficientSize, ML_DEVICE);
    cu_copy_H2D_async(pme->gpu->coefficients, coefficients_h, coefficientSize, pme->gpu->pmeStream);
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

static gmx_bool debugMemoryPrint = false;

void PMEMemoryFree(PMEDataID id, MemLocType location)
{
    cudaError_t stat;
    size_t i = location * PME_ID_END_INVALID + id;
    if (PMEStoragePointers[i])
    {
        if (debugMemoryPrint)
            printf("free! %p %d %d\n", PMEStoragePointers[i], id, location);
        if (location == ML_DEVICE)
        {
            stat = cudaFree(PMEStoragePointers[i]);
            CU_RET_ERR(stat, "PME cudaFree error");
        }
        else
        {
            stat = cudaFreeHost(PMEStoragePointers[i]);
            CU_RET_ERR(stat, "PME cudaFreeHost error");
        }
        PMEStoragePointers[i] = NULL;
    }
}

void *PMEMemoryFetch(PMEDataID id, size_t size, MemLocType location)
{
    // size == 0 => just return a current pointer

    //yupinov grid resize mistake!
    cudaError_t stat = cudaSuccess;
    size_t i = location * PME_ID_END_INVALID + id;

    if ((PMEStorageSizes[i] > 0) && (size > 0) && (size > PMEStorageSizes[i]))
        printf("asked to realloc %lu into %lu with ID %d\n", PMEStorageSizes[i], size, id);

    if (PMEStorageSizes[i] < size) // delete
    {
        PMEMemoryFree(id, location);
        if (size > 0)
        {
            if (debugMemoryPrint)
                printf("asked to alloc %lu", size);
            size = size * 1.02; // slight overalloc for no apparent reason
            if (debugMemoryPrint)
                printf(", actually allocating %lu\n", size);
            if (location == ML_DEVICE)
            {
                stat = cudaMalloc((void **)&PMEStoragePointers[i], size);
                CU_RET_ERR(stat, "PME cudaMalloc error");
            }
            else
            {
                unsigned int allocFlags = cudaHostAllocDefault;
                //allocFlags |= cudaHostAllocWriteCombined;
                //yupinov try cudaHostAllocWriteCombined for almost-constant global memory? do I even have that?
                // yes, I do: coordinates/coefficients and thetas/dthetas. should be helpful for spread being overwhelmed by L2 cache!
                stat = cudaHostAlloc((void **)&PMEStoragePointers[i], size, allocFlags);
                CU_RET_ERR(stat, "PME cudaHostAlloc error");
            }
            PMEStorageSizes[i] = size;
        }
    }
    return PMEStoragePointers[i];
}

void PMECopyConstant(const void *dest, void const *src, size_t size, cudaStream_t s)
{
    assert(s != 0);
    cudaError_t stat = cudaMemcpyToSymbolAsync(dest, src, size, 0, cudaMemcpyHostToDevice, s);
    CU_RET_ERR(stat, "PME cudaMemcpyToSymbolAsync");
}

