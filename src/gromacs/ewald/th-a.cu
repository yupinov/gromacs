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
    if (th_size[i] < size || size == 0)
    {
        if (th_p[i])
        {
            if (th_a_print)
                fprintf(stderr, "free! %p\n", th_p[i]);
            if (loc == TH_LOC_CUDA)
            {
                stat = cudaFree(th_p[i]);
                CU_RET_ERR(stat, "cudaFree th error");
            }
            else
            {
                delete[] (T *) th_p[i];
            }
            th_p[i] = NULL;
        }
        if (size > 0)
        {
            size = size * 2 + 16;
            if (th_a_print)
                fprintf(stderr, "alloc! %d\n", size);
            if (loc == TH_LOC_CUDA)
            {
                stat = cudaMalloc((void **) &th_p[i], size);
                CU_RET_ERR(stat, "cudaMalloc th error");
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


real *th_a(th_id id, int thread, int size, th_loc loc) {
  return th_t<real>(id, thread, size, loc);
}


int *th_i(th_id id, int thread, int size, th_loc loc) {
  return th_t<int>(id, thread, size, loc);
}

void th_cpy(void *dest, void *src, int size, th_loc dest_loc) {
  if (dest_loc == TH_LOC_CUDA)
  {
    cudaError_t stat = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
    CU_RET_ERR(stat, "cudaMemcpyHostToDevice th error");
  }
  else
  {
    cudaError_t stat = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
    CU_RET_ERR(stat, "cudaMemcpyDeviceToHost th error");
  }
}

