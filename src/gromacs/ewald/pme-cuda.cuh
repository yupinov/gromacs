#ifndef PME_CUDA_H
#define PME_CUDA_H

#include "pme-internal.h"

#include "pme-timings.cuh"

#include "gromacs/gpu_utils/cudautils.cuh"

#include <vector>

//yupinov grid indices

#define PME_USE_TEXTURES 1
// using textures instead of global memory

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
#endif

// identifiers for PME data stored on GPU
enum PMEDataID
{
    PME_ID_THETA = 0,
    PME_ID_DTHETA,

    // grids
    PME_ID_REAL_GRID, // functions as pme_grid with overlap and as fftgrid
    PME_ID_COMPLEX_GRID, // used only for out-of-place cuFFT, functions as cfftgrid

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

    PME_ID_BSP_MOD_XX, PME_ID_BSP_MOD_YY, PME_ID_BSP_MOD_ZZ,

    // solve_lj
    PME_ID_ENERGY,
    PME_ID_VIRIAL,

    // solve
    PME_ID_ENERGY_AND_VIRIAL,

    // end
    PME_ID_END_INVALID
};

enum MemLocType
{
    ML_HOST = 0, ML_DEVICE, ML_END_INVALID
};

// PME GPU structures

// spread/solve/gather pme inverted box

// CAREFUL: the box is transposed as compared to the original pme->recipbox
// basically, spread uses matrix columns (while solve and gather use rows)
// that's the reason why I transposed the box initially
// maybe swap it the other way around?
//yupinov - check on triclinic!

struct pme_gpu_recipbox_t
{
    float3 box[DIM];
};

// wrap/unwrap overlap zones
struct pme_gpu_overlap_t
{
#define OVERLAP_ZONES 7
    int2 overlapSizes[OVERLAP_ZONES];
    int overlapCellCounts[OVERLAP_ZONES];
};

struct pme_gpu_const_parameters
{
    // sizes
    rvec nXYZ;
};

struct gmx_pme_cuda_t
{
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

    // some other permanent settings set on init

    gmx_bool doOutOfPlaceFFT; // if true, then an additional grid of the same size is used for R2C/solve/C2R

    gmx_bool doTime; // enable timing using CUDA events

    gmx_bool useTextureObjects; // if false, then use references

#if !PME_EXTERN_CMEM
    // constant structures for arguments
    pme_gpu_recipbox_t recipbox;
    pme_gpu_overlap_t overlap;
#endif


    gmx_device_info_t *deviceInfo;

    pme_gpu_timing timingEvents[PME_GPU_STAGES];

    gmx_parallel_3dfft_gpu_t *pfft_setup_gpu;


    // internal host/device pointers storage
    std::vector<size_t> StorageSizes;
    std::vector<void *> StoragePointers;


    // some device pointers/objects below - they are assigned from the PMEStoragePointers!

    // spline calculation
    // fractional shifts (pme->fsh*)
    real *fshArray;
    // indices (pme->nn*)
    int *nnArray;

    // real grid - used everywhere
    real *grid;
    // complex grid - used in R2C/solve/C2R
    // if we're using inplace cuFFT, then it's the same pointer as grid!
    t_complex *fourierGrid;

    // solve
    // 6 virial components, energy => 7 elements
    real *energyAndVirial;
    size_t energyAndVirialSize; //bytes

    // gather
    // forces
    real *forces;


    // forces and coordinates should be shared with nonbondeds!
    float3 *coordinates;
    real *coefficients;

    pme_gpu_const_parameters constants;
};

// allocate memory; size == 0 => just fetch the current pointer
void *PMEMemoryFetch(gmx_pme_t *pme, PMEDataID id, size_t size, MemLocType location);
// deallocate memory
void PMEMemoryFree(gmx_pme_t *pme, PMEDataID id, MemLocType location);

void PMECopyConstant(const void *dest, const void *src, size_t size, cudaStream_t s); //H2D only
#endif
