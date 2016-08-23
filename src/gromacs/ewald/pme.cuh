/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#ifndef PME_CUDA_H
#define PME_CUDA_H

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "pme-internal.h"  //?
#include "pme-timings.cuh" //?


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
    PME_ID_REAL_GRID,    // functions as pme_grid with overlap and as fftgrid
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

// wrap/unwrap overlap zones
struct pme_gpu_overlap_t
{
#define OVERLAP_ZONES 7
    int2 overlapSizes[OVERLAP_ZONES];
    int  overlapCellCounts[OVERLAP_ZONES];
};

/* A structure for storing common constants accessed within GPU kernels by value.
 * Some things are really constant, some things change every step => more splitting?
 * Ideally, as there are many PME kernels, this should not be a kernel parameter,
 * but rather a constant memory copied once per step.
 */
struct pme_gpu_const_parameters
{
    /* Reciprocal box */
    //yupinov specify column or row
    float3 recipbox[DIM];
    /* Grid sizes */
    int3   localGridSize;
    float3 localGridSizeFP;
    int3   localGridSizePadded; /* padding includes (order - 1) overlap and possibly some alignment in Z? */
    /* Number of local atoms */
    int    nAtoms;

    /* Solving parameters - maybe they should be in a separate structure,
     * as we likely won't use GPU solve much in multi-rank PME? */
    float volume;      /* The unit cell volume */
    float ewaldFactor; /* (M_PI / ewaldCoeff)^2 */
    float elFactor;    /* ONE_4PI_EPS0 / pme->epsilon_r */
};

/* The main PME GPU structure, included in the PME CPU structure by pointer */
struct gmx_pme_cuda_t
{
    cudaStream_t pmeStream;

    // synchronization events
    cudaEvent_t syncEnerVirD2H;    // energy and virial have already been calculated in pme-solve, and have been copied to host
    cudaEvent_t syncForcesD2H;     // forces have already been calculated in pme-gather, and have been copied to host
    cudaEvent_t syncSpreadGridD2H; // the grid has been copied to the host after the spreading for CPU FFT
    cudaEvent_t syncSolveGridD2H;  // the grid has been copied to the host after the solve for CPU FFT

    // some other permanent settings set on init

    // gmx_bool bGPUSpread;
    // spread being on a GPU is a given - it's the main effort

    gmx_bool bGPUSolve;                                             /* Are we doing the solve stage on the GPU? */

    gmx_bool bGPUGather;                                            /* Are we doing the gather stage on the GPU? */

    gmx_bool bGPUFFT;                                               /* Are we using cuFFT as well? Currently only for a single rank */

    gmx_bool bGPUSingle;                                            /* Are we using the single GPU rank? A convenience variable */

    gmx_bool bOutOfPlaceFFT;                                        /* If true, then an additional grid of the same size is used for R2C/solve/C2R */

    gmx_bool bTiming;                                               /* Enable timing using CUDA events */

    gmx_bool useTextureObjects; /* If false, then use references */ //unused!

    // constant structures for arguments
    pme_gpu_overlap_t             overlap;
    pme_gpu_const_parameters      constants;

    gmx_device_info_t            *deviceInfo;

    std::vector<pme_gpu_timing *> timingEvents;

    gmx_parallel_3dfft_gpu_t     *pfft_setup_gpu;


    // internal host/device pointers storage
    std::vector<size_t> StorageSizes;
    std::vector<void *> StoragePointers;


    // some device pointers/objects below - they are assigned from the PMEStoragePointers

    // spline calculation
    // fractional shifts (pme->fsh*)
    real *fshArray;
    // indices (pme->nn*)
    int  *nnArray;

    // real grid - used everywhere
    real *grid;
    // complex grid - used in R2C/solve/C2R
    // if we're using inplace cuFFT, then it's the same pointer as grid!
    t_complex *fourierGrid;

    // solve
    // 6 virial components, energy => 7 elements
    real  *energyAndVirial;
    size_t energyAndVirialSize; // bytes

    // gather
    // forces
    real *forces;

    // forces and coordinates should be shared with nonbondeds!
    float3 *coordinates;
    real   *coefficients;
};

// allocate memory; size == 0 => just fetch the current pointer
void *PMEMemoryFetch(const gmx_pme_t *pme, PMEDataID id, size_t size, MemLocType location);


// dumping all the CUDA-specific PME functions here...

// copies the bspline moduli to the device (used in PME solve)
void pme_gpu_copy_bspline_moduli(const gmx_pme_t *pme);

/*! \brief Copies the charges to the GPU */
void pme_gpu_copy_charges(const gmx_pme_t *pme);


void gmx_parallel_3dfft_destroy_gpu(const gmx_parallel_3dfft_gpu_t &pfft_setup);



// copies the nn and fsh to the device (used in PME spread(spline))
void pme_gpu_copy_calcspline_constants(const gmx_pme_t *pme);

// clearing
void pme_gpu_clear_grid(const gmx_pme_t *pme, const int grid_index);
void pme_gpu_clear_energy_virial(const gmx_pme_t *pme, const int grid_index);

// allocating
void pme_gpu_alloc_grids(const gmx_pme_t *pme, const int grid_index);
void pme_gpu_alloc_energy_virial(const gmx_pme_t *pme, const int grid_index);
void pme_gpu_alloc_gather_forces(const gmx_pme_t *pme);

void gmx_parallel_3dfft_init_gpu(gmx_parallel_3dfft_gpu_t *pfft_setup,
                                 ivec                      ndata,
                                 const gmx_pme_t          *pme);

#endif
