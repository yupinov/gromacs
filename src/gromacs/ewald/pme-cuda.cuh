#ifndef PME_CUDA_H
#define PME_CUDA_H

#include "pme-internal.h"

struct gmx_pme_cuda_t
{
    cudaStream_t pmeStream;
    gmx_bool keepGPUDataBetweenSpreadAndR2C; //yupinov BetweenSplineAndSpread?
    //yupinov should be same as keepGPUDataBetweenC2RAndGather ? or what do I do wit hdthetas?
    gmx_bool keepGPUDataBetweenR2CAndSolve;
    gmx_bool keepGPUDataBetweenSolveAndC2R;
    gmx_bool keepGPUDataBetweenC2RAndGather;
    //yupinov init
    //keep those as params in the th storage
};
//yupinov dealloc
//yupinov grid indices with tags?

#define PME_CUFFT_INPLACE
// comment this to enable out-of-place cuFFT
// it requires a separate complex grid, seems to be virtually the same performance-wise

#define PME_GPU_TIMINGS
// should replace this to respect other GPU timings' variables
// comment this to disable PME timing function bodies

static const bool PME_SKIP_ZEROES = false;
// skipping particles with zero charges on a CPU side
// for now only done in gather, should be done in spread and memorized
// seems like a total waste of time! but what if we do it once at each NS?


// identifiers for PME data stored on GPU
enum PMEDataID
{
    PME_ID_THETA = 1, PME_ID_DTHETA, PME_ID_FRACTX, PME_ID_COEFFICIENT,

    //yupinov fix unnecesary memory usage
    PME_ID_REAL_GRID, //this is pme_grid and it has overlap
#ifndef PME_CUFFT_INPLACE
    PME_ID_COMPLEX_GRID, //this is cfftgrid
#endif
    PME_ID_I0, PME_ID_J0, PME_ID_K0,

    PME_ID_THX, PME_ID_THY, PME_ID_THZ, //only used on host now

    // interpol_idx
    PME_ID_FSH,
    PME_ID_NN,
    PME_ID_XPTR,

    PME_ID_IDXPTR, //yupinov added - a duplicate of PME_ID_I0, PME_ID_J0, PME_ID_K0,
    PME_ID_F,
    PME_ID_NONZERO_INDICES,
    PME_ID_DTHX, PME_ID_DTHY, PME_ID_DTHZ,
    PME_ID_BSP_MOD_MINOR, PME_ID_BSP_MOD_MAJOR, PME_ID_BSP_MOD_MIDDLE,
    PME_ID_ENERGY,
    PME_ID_VIRIAL,

    PME_ID_END_INVALID
};

#ifdef PME_CUFFT_INPLACE
#define PME_ID_COMPLEX_GRID PME_ID_REAL_GRID
#endif

enum MemLocType
{
    ML_HOST, ML_DEVICE, ML_END_INVALID
};

//yupinov - look into ML_HOST being under-used

real *PMEFetchRealArray(PMEDataID id, int unusedTag, int size, MemLocType location);
int *PMEFetchIntegerArray(PMEDataID id, int unusedTag, int size, MemLocType location);
t_complex *PMEFetchComplexArray(PMEDataID id, int unusedTag, int size, MemLocType location);
//yupinov warn on wrong param

void PMECopy(void *dest, void *src, int size, MemLocType destination, cudaStream_t s); //yupinov alloc as well

int *PMEFetchAndCopyIntegerArray(PMEDataID id, int unusedTag, void *src, int size, MemLocType location, cudaStream_t s);
real *PMEFetchAndCopyRealArray(PMEDataID id, int unusedTag, void *src, int size, MemLocType location, cudaStream_t s);
t_complex *PMEFetchAndCopyComplexArray(PMEDataID id, int unusedTag, void *src, int size, MemLocType location, cudaStream_t s);

#endif
