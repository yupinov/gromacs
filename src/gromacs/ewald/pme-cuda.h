#ifndef PMECUDAH
#define PMECUDAH
#include "pme-internal.h"
struct gmx_pme_cuda_t
{
    cudaStream_t pmeStream;
    gmx_bool keepGPUDataBetweenSpreadAndR2C; //yupinov BetweenSplineAndSpread?
    gmx_bool keepGPUDataBetweenR2CAndSolve;
    gmx_bool keepGPUDataBetweenSolveAndC2R;
    gmx_bool keepGPUDataBetweenC2RAndGather;
    //yupinov init
    //keep those as params in the th storage
};

//yupinov dealloc
//yupinov grid indices?

#define PME_CUFFT_INPLACE
//yupinov - seems to perform same?

#endif
