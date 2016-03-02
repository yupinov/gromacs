#include <vector>
#include <stdio.h>
#include "th-a.cuh"
#include "gromacs/gpu_utils/cudautils.cuh"

static std::vector<int> th_size(TH_LOC_END * TH_ID_END * TH);
static std::vector<void *> th_p(TH_LOC_END * TH_ID_END * TH);

static const bool th_a_print = false;

template <typename T>
T *th_t(th_id id, int thread, int size, th_loc loc)
{
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
                th_p[i] = new T[size / sizeof(T)];
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
T *th_t_cpy(th_id id, int thread, void *src, int size, th_loc loc)
{
    T *result = th_t<T>(id, thread, size, loc);
    th_cpy(result, src, size, loc);
    return result;
}

t_complex *th_c_cpy(th_id id, int thread, void *src, int size, th_loc loc)
{
    return th_t_cpy<t_complex>(id, thread, src, size, loc);
}

real *th_a_cpy(th_id id, int thread, void *src, int size, th_loc loc)
{
    return th_t_cpy<real>(id, thread, src, size, loc);
}

void th_cpy(void *dest, void *src, int size, th_loc dest_loc) //yupinov move everything onto this function - or not
{
    if (dest_loc == TH_LOC_CUDA)
    {
        cudaError_t stat = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
        CU_RET_ERR(stat, "PME cudaMemcpyHostToDevice error");
    }
    else
    {
        cudaError_t stat = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
        CU_RET_ERR(stat, "PME cudaMemcpyDeviceToHost error");
    }
}

