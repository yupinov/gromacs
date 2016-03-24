#include <vector>
#include <stdio.h>
#include "gromacs/gpu_utils/cudautils.cuh"

#include "pme-cuda.cuh"

#include <assert.h>

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

void pme_gpu_init(gmx_pme_gpu_t **pmeGPU)
{
    *pmeGPU = new gmx_pme_gpu_t;
    cudaError_t stat;
//yupinov dealloc@

#if GMX_CUDA_VERSION >= 5050
    int highest_priority;
    int lowest_priority;
    stat = cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
    CU_RET_ERR(stat, "PME cudaDeviceGetStreamPriorityRange failed");
    stat = cudaStreamCreateWithPriority(&(*pmeGPU)->pmeStream,
                                            //cudaStreamNonBlocking,
                                            cudaStreamDefault,
                                            highest_priority);
    //yupinov: fighting with nbnxn non-local for highest priority - check on MPI!
    CU_RET_ERR(stat, "cudaStreamCreateWithPriority on PME stream failed");
#else
    stat = cudaStreamCreate(&(*pme)->pmeStream);
    CU_RET_ERR(stat, "PME cudaStreamCreate error");
#endif

    pme_gpu_update_flags(*pmeGPU, false, false, false, false);
}

#define MAXTAGS 1

static std::vector<int> PMEStorageSizes(ML_END_INVALID * PME_ID_END_INVALID * MAXTAGS);
static std::vector<void *> PMEStoragePointers(ML_END_INVALID * PME_ID_END_INVALID * MAXTAGS);

static bool debugMemoryPrint = false;

template <typename T>
T *PMEFetch(PMEDataID id, int unusedTag, int size, MemLocType location)
{
    //yupinov grid resize mistake!
    assert(unusedTag == 0);
    cudaError_t stat;
    int i = (location * PME_ID_END_INVALID + id) * MAXTAGS + unusedTag;

    if ((PMEStorageSizes[i] > 0) && (size > 0) && (size > PMEStorageSizes[i]))
        printf("asked to realloc %d into %d with ID %d\n", PMEStorageSizes[i], size, id);

    if (PMEStorageSizes[i] < size || size == 0) //delete
    {
        if (PMEStoragePointers[i])
        {
            if (debugMemoryPrint)
                fprintf(stderr, "free! %p\n", PMEStoragePointers[i]);
            if (location == ML_DEVICE)
            {
                stat = cudaFree(PMEStoragePointers[i]);
                CU_RET_ERR(stat, "PME cudaFree error");
            }
            else
            {
                delete[] (T *) PMEStoragePointers[i];
            }
            PMEStoragePointers[i] = NULL;
        }
        if (size > 0)
        {
            if (debugMemoryPrint)
                printf("asked to alloc %d", size);
            size = size * 1.02; //yupinov overalloc
            if (debugMemoryPrint)
                printf(", actually allocating %d\n", size);
            if (location == ML_DEVICE)
            {
                stat = cudaMalloc((void **) &PMEStoragePointers[i], size);
                CU_RET_ERR(stat, "PME cudaMalloc error");
            }
            else
            {
                PMEStoragePointers[i] = new T[size / sizeof(T)]; //yupinov cudaHostMalloc?
            }
            PMEStorageSizes[i] = size;
        }
    }
    return (T *) PMEStoragePointers[i];
}

real *PMEFetchRealArray(PMEDataID id, int unusedTag, int size, MemLocType location)
{
    return PMEFetch<real>(id, unusedTag, size, location);
}

t_complex *PMEFetchComplexArray(PMEDataID id, int unusedTag, int size, MemLocType location)
{
    return PMEFetch<t_complex>(id, unusedTag, size, location);
}

int *PMEFetchIntegerArray(PMEDataID id, int unusedTag, int size, MemLocType location)
{
    return PMEFetch<int>(id, unusedTag, size, location);
}

template <typename T>
T *PMEFetchAndCopy(PMEDataID id, int unusedTag, void *src, int size, MemLocType location, cudaStream_t s)
{
    T *result = PMEFetch<T>(id, unusedTag, size, location);
    PMECopy(result, src, size, location, s);
    return result;
}

t_complex *PMEFetchAndCopyComplexArray(PMEDataID id, int unusedTag, void *src, int size, MemLocType location, cudaStream_t s)
{
    return PMEFetchAndCopy<t_complex>(id, unusedTag, src, size, location, s);
}

real *PMEFetchAndCopyRealArray(PMEDataID id, int unusedTag, void *src, int size, MemLocType location, cudaStream_t s)
{
    return PMEFetchAndCopy<real>(id, unusedTag, src, size, location, s);
}

int *PMEFetchAndCopyIntegerArray(PMEDataID id, int unusedTag, void *src, int size, MemLocType location, cudaStream_t s)
{
    return PMEFetchAndCopy<int>(id, unusedTag, src, size, location, s);
}

void PMECopy(void *dest, void *src, int size, MemLocType destination, cudaStream_t s) //yupinov move everything onto this function - or not
{
    if (destination == ML_DEVICE)
    {
        //cudaError_t stat = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
        cudaError_t stat = cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, s);
        CU_RET_ERR(stat, "PME cudaMemcpyHostToDevice error");
    }
    else
    {
        //cudaError_t stat = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
        cudaError_t stat = cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, s);
        CU_RET_ERR(stat, "PME cudaMemcpyDeviceToHost error");
    }
}

