#ifndef PMECUDAH
#define PMECUDAH
#include "pme-internal.h"
struct gmx_pme_cuda_t
{
    cudaStream_t pmeStream;
};

//yupinov dealloc

//#define PME_CUFFT_INPLACE
//yupinov - seems to perform worse?

#endif
