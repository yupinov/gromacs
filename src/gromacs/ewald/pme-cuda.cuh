#ifndef PME_CUDA_H
#define PME_CUDA_H

#include "gromacs/gpu_utils/cudautils.cuh"
#include "pme-internal.h"
#include "pme-timings.cuh"

#include <vector>

#define PME_USE_TEXTURES 1
// using textures instead of global memory

// particles per block and block sizes should also be here....

// the hierarchy of the global spline data was for some silly reason order -> particle -> dimension
//#define PME_SPLINE_ORDER_STRIDE (particlesPerBlock * DIM)
//#define PME_SPLINE_PARTICLE_STRIDE (DIM)

// particles should be on top, so now it's particle -> order -> dimension
// that means switching the particle and order indices
#define PME_SPLINE_ORDER_STRIDE (DIM)
#define PME_SPLINE_PARTICLE_STRIDE (DIM * order)

#define PME_SPLINE_THETA_STRIDE 1
// should be 2 for float2 theta/dtheta storage

#define PME_SPREADGATHER_BLOCK_DATA_SIZE (particlesPerBlock * DIM)

#define PARTICLES_PER_WARP (warp_size / order / order)
// there is some redundancy here!
// what do I do when order > 5?

//yupinov - document spline param layout (2 particles -> order -> dim -> particle 1, 2)

// internal identifiers for PME data stored on GPU and host
enum PMEDataID
{
    PME_ID_THETA = 0,
    PME_ID_DTHETA,

    // grids
    PME_ID_REAL_GRID, // functions as pme_grid with overlap and as fftgrid
    PME_ID_COMPLEX_GRID, // used only for out-of-place cuFFT, functions as cfftgrid

    // spread (spline)
    PME_ID_FSH,
    PME_ID_NN,

    // spread
    PME_ID_XPTR,
    PME_ID_COEFFICIENT, // atc->coefficient

    // gather
    PME_ID_FORCES,

    // spread and gather
    PME_ID_IDXPTR, // grid indices as in atc->idx

    // solve_lj
    PME_ID_ENERGY,
    PME_ID_VIRIAL,

    // solve
    PME_ID_ENERGY_AND_VIRIAL,
    PME_ID_BSP_MOD_XX, PME_ID_BSP_MOD_YY, PME_ID_BSP_MOD_ZZ, // B-spline moduli

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
// maybe swap it the other way around?

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
    // grid sizes
    rvec nXYZ;
    // number of local particles
    int nAtoms;
};

// the main PME GPU structure
struct gmx_pme_cuda_t
{
    cudaStream_t pmeStream;

    // synchronization events
    cudaEvent_t syncEnerVirD2H; // energy and virial have already been calculated in pme-solve, and have been copied to host
    cudaEvent_t syncForcesD2H;  // forces have already been calculated in pme-gather, and have been copied to host
    cudaEvent_t syncSpreadGridD2H; // the grid has been copied to the host after the spreading for CPU FFT
    cudaEvent_t syncSolveGridD2H; // the grid has been copied to the host after the solve for CPU FFT

    // some other permanent settings set on init

    // gmx_bool bGPUSpread;
    // spread being on a GPU is a given - it's the main effort

    gmx_bool bGPUSolve; /* Are we doing the solve stage on the GPU? */

    gmx_bool bGPUGather; /* Are we doing the gather stage on the GPU? */

    gmx_bool bGPUFFT; /* Are we using cuFFT as well? Currently only for a single rank */

    gmx_bool bGPUSingle; /* Are we using the single GPU rank? A convenience variable */

    gmx_bool bOutOfPlaceFFT; /* If true, then an additional grid of the same size is used for R2C/solve/C2R */

    gmx_bool bTiming; /* Enable timing using CUDA events */

    gmx_bool useTextureObjects; /* If false, then use references */

    // constant structures for arguments
    pme_gpu_recipbox_t recipbox;
    pme_gpu_overlap_t overlap;

    gmx_device_info_t *deviceInfo;

    std::vector<pme_gpu_timing *> timingEvents;

    gmx_parallel_3dfft_gpu_t *pfft_setup_gpu;


    // internal host/device pointers storage
    std::vector<size_t> StorageSizes;
    std::vector<void *> StoragePointers;


    // some device pointers/objects below - they are assigned from the PMEStoragePointers

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
    size_t energyAndVirialSize; // bytes

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

void PMECopyConstant(const void *dest, const void *src, size_t size, cudaStream_t s); // H2D

#endif
