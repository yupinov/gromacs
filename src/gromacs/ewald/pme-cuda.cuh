#ifndef PME_CUDA_H
#define PME_CUDA_H

#include "pme-internal.h"

#include "pme-timings.cuh"

//yupinov dealloc
//yupinov grid indices with tags?

// device constants
// wrap/unwrap overlap zones


// spread/solve/gather pme inverted box

// CAREFUL: the box is transposed as compared to the original pme->recipbox
// basically, spread uses matrix columns (while solve and gather use rows)
// that's the reason why I transposed the box initially
// maybe swap it the otehr way around?
//yupinov - check on triclinic!
//yupinov - load them once in GPU init! check if loaded

#define PME_CUFFT_INPLACE 1
// comment this to enable out-of-place cuFFT
// it requires a separate complex grid, seems to be virtually the same performance-wise

#define PME_GPU_TIMINGS 1
// comment this to disable PME timing function bodies
// should replace this to respect other GPU timings' variables


static const bool PME_SKIP_ZEROES = false;
// broken
// skipping particles with zero charges on a CPU side
// for now only done in gather, should be done in spread and memorized
// seems like a total waste of time! but what if we do it once at each NS?


#define PME_EXTERN_CMEM 0
// constants as extern instead of arguments -> needs CUDA_SEPARABLE_COMPILATION which is off by default

#if PME_EXTERN_CMEM
#error "Unfinished separable compilation implementation"

// spread/solve/gather
extern __constant__ __device__ float3 RECIPBOX[3];
// wrap/unwrap
#define OVERLAP_ZONES 7
extern __constant__ __device__ int2 OVERLAP_SIZES[OVERLAP_ZONES];
extern __constant__ __device__ int OVERLAP_CELLS_COUNTS[OVERLAP_ZONES];
#else

struct pme_gpu_recipbox_t
{
    float3 box[DIM];
};

struct pme_gpu_overlap_t
{
#define OVERLAP_ZONES 7
    int2 overlapSizes[OVERLAP_ZONES];
    int overlapCellCounts[OVERLAP_ZONES];
};
#endif

struct pme_gpu_timing;

struct gmx_pme_cuda_t
{
    // a stream where everything should happen
    cudaStream_t pmeStream;

    // synchronization events
    cudaEvent_t syncEnerVirH2D; // energy and virial have already been calculated in pme-solve, and have been copied to host
    cudaEvent_t syncForcesH2D;  // forces have already been calculated in pme-gather, and have been copied to host

    // crude data-keeping flags
    gmx_bool keepGPUDataBetweenSpreadAndR2C; //yupinov BetweenSplineAndSpread?
    //yupinov should be same as keepGPUDataBetweenC2RAndGather ? or what do I do wit hdthetas?
    gmx_bool keepGPUDataBetweenR2CAndSolve;
    gmx_bool keepGPUDataBetweenSolveAndC2R;
    gmx_bool keepGPUDataBetweenC2RAndGather;

    //keep those as params in the th storage
#if !PME_EXTERN_CMEM
    // constant structures for arguments
    pme_gpu_recipbox_t recipbox;
    pme_gpu_overlap_t overlap;
#endif

    pme_gpu_timing timingEvents[PME_GPU_STAGES];

    // device pointers/obejcts below

    // spline calculation
    // fractional shifts (pme->fsh*)
    real *fshArray;
    // indices (pme->nn*)
    int *nnArray;

    // grid - used everywhere
    real *grid;

    // gather
    // forces
    real *forces;


    //forces and coordinates should be shared with nonbondeds!

};


// identifiers for PME data stored on GPU
enum PMEDataID
{
    PME_ID_THETA = 1,
    PME_ID_DTHETA,

    PME_ID_REAL_GRID, //this is pme_grid and it has overlap
#if !PME_CUFFT_INPLACE
    PME_ID_COMPLEX_GRID, //this is cfftgrid
#endif

    // only used on host in gather now
    PME_ID_THX, PME_ID_THY, PME_ID_THZ,
    PME_ID_DTHX, PME_ID_DTHY, PME_ID_DTHZ,

    // interpol/spline
    PME_ID_FSH,
    PME_ID_NN,

    // spread
    PME_ID_XPTR,

    // gather
    PME_ID_FORCES,
    PME_ID_NXYZ,
    PME_ID_NONZERO_INDICES, // compacted data indices

    // spread and gather

    PME_ID_IDXPTR, // grid indices as in atc->idx
    //PME_ID_I0, PME_ID_J0, PME_ID_K0, // same, but sorted (spearate XX, YY, ZZ arrays)

    PME_ID_COEFFICIENT, //atc->coefficient

    PME_ID_BSP_MOD_MINOR, PME_ID_BSP_MOD_MAJOR, PME_ID_BSP_MOD_MIDDLE,

    // solve_lj
    PME_ID_ENERGY,
    PME_ID_VIRIAL,

    // solve
    PME_ID_ENERGY_AND_VIRIAL,

    // end
    PME_ID_END_INVALID
};

#if PME_CUFFT_INPLACE
#define PME_ID_COMPLEX_GRID PME_ID_REAL_GRID
#endif

enum MemLocType
{
    ML_HOST, ML_DEVICE, ML_END_INVALID
};

// ML_HOST under-used; what about pinning memory?

// all sizes here are in bytes

real *PMEFetchRealArray(PMEDataID id, int unusedTag, int size, MemLocType location);
int *PMEFetchIntegerArray(PMEDataID id, int unusedTag, int size, MemLocType location);
t_complex *PMEFetchComplexArray(PMEDataID id, int unusedTag, int size, MemLocType location);
//yupinov warn on wrong param

void PMECopy(void *dest, void *src, int size, MemLocType destination, cudaStream_t s, gmx_bool sync = false); //yupinov alloc as well
void PMECopyConstant(const void *dest, const void *src, size_t size, cudaStream_t s); //H2D only

int *PMEFetchAndCopyIntegerArray(PMEDataID id, int unusedTag, void *src, int size, MemLocType location, cudaStream_t s);
real *PMEFetchAndCopyRealArray(PMEDataID id, int unusedTag, void *src, int size, MemLocType location, cudaStream_t s, gmx_bool sync = false);
t_complex *PMEFetchAndCopyComplexArray(PMEDataID id, int unusedTag, void *src, int size, MemLocType location, cudaStream_t s);

int PMEGetAllocatedSize(PMEDataID id, int unusedTag, MemLocType location);

#endif
