#include <vector>
#include <stdio.h>
#include "th-a.cuh"
#include "gromacs/gpu_utils/cudautils.cuh"


#include "pme-cuda.h"
void pme_gpu_init(gmx_pme_gpu_t **pme)
{
    *pme = new gmx_pme_gpu_t;
    cudaError_t stat;
//yupinov dealloc@

// there are 3 situations (all tested with a single rank):
// no priority support => big hole between nbnxn and spread3
// creating PME stream with no priority => small hole, only memcpy hidden, memset is synced (?)
// creating PME steram with highest priority (out of 2, lol) => actually works in parallel
// but still, a lot of spread D2H even in the last case....
// in short, priority doesn't hurt, but only shows up on Tesla
#if GMX_CUDA_VERSION >= 5050
    int highest_priority;
    int lowest_priority;
    stat = cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
    CU_RET_ERR(stat, "cudaDeviceGetStreamPriorityRange failed");
    stat = cudaStreamCreateWithPriority(&(*pme)->pmeStream,
                                            //cudaStreamNonBlocking,
                                            cudaStreamDefault, //yupinov why not ?
                                            highest_priority);
    //yupinov: fighting with nbnxn non-local for highest priority - check on MPI!
    CU_RET_ERR(stat, "cudaStreamCreateWithPriority on PME stream failed");
#else
    stat = cudaStreamCreate(&(*pme)->pmeStream);
    CU_RET_ERR(stat, "PME cudaStreamCreate error");
#endif
}


static std::vector<int> th_size(TH_LOC_END * TH_ID_END * TH);
static std::vector<void *> th_p(TH_LOC_END * TH_ID_END * TH);

static bool th_a_print = false;

template <typename T>
T *th_t(th_id id, int thread, int size, th_loc loc)
{
    //yupinov different size mistake!
    cudaError_t stat;
    int i = (loc * TH_ID_END + id) * TH + thread;
    if (th_size[i] < size || size == 0) //delete
    {
        if (th_p[i])
        {
            if (th_a_print)
                fprintf(stderr, "free! %p\n", th_p[i]);
            if (loc == TH_LOC_CUDA)
            {
                stat = cudaFree(th_p[i]);
                CU_RET_ERR(stat, "PME cudaFree error");
            }
            else
            {
                delete[] (T *) th_p[i];
            }
            th_p[i] = NULL;
        }
        if (size > 0)
        {
            if (th_size[i] != 0)
                printf("asked to realloc %d into %d with ID %d\n", th_size[i], size, id);
            if (th_a_print)
                printf("asked to alloc %d", size);
            size = size * 1.02; //yupinov overalloc
            if (th_a_print)
                printf(", actually allocating %d\n", size);
            if (loc == TH_LOC_CUDA)
            {
                stat = cudaMalloc((void **) &th_p[i], size);
                CU_RET_ERR(stat, "PME cudaMalloc error");
            }
            else
            {
                th_p[i] = new T[size / sizeof(T)]; //yupinov cudaHostMalloc?
            }
            th_size[i] = size;
        }
    }
    return (T *) th_p[i];
}

real *th_a(th_id id, int thread, int size, th_loc loc)
{
    return th_t<real>(id, thread, size, loc);
}

t_complex *th_c(th_id id, int thread, int size, th_loc loc)
{
    return th_t<t_complex>(id, thread, size, loc);
}

int *th_i(th_id id, int thread, int size, th_loc loc)
{
    return th_t<int>(id, thread, size, loc);
}

template <typename T>
T *th_t_cpy(th_id id, int thread, void *src, int size, th_loc loc, cudaStream_t s)
{
    T *result = th_t<T>(id, thread, size, loc);
    th_cpy(result, src, size, loc, s);
    return result;
}

t_complex *th_c_cpy(th_id id, int thread, void *src, int size, th_loc loc, cudaStream_t s)
{
    return th_t_cpy<t_complex>(id, thread, src, size, loc, s);
}

real *th_a_cpy(th_id id, int thread, void *src, int size, th_loc loc, cudaStream_t s)
{
    return th_t_cpy<real>(id, thread, src, size, loc, s);
}

int *th_i_cpy(th_id id, int thread, void *src, int size, th_loc loc, cudaStream_t s)
{
    return th_t_cpy<int>(id, thread, src, size, loc, s);
}

void th_cpy(void *dest, void *src, int size, th_loc dest_loc, cudaStream_t s) //yupinov move everything onto this function - or not
{
    if (dest_loc == TH_LOC_CUDA)
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

