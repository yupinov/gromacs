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

/*! \internal \file
 * \brief This file defines the PME CUDA data structures,
 * various compile-time constants shared among the PME CUDA kernels,
 * and also names a bunch of the PME CUDA memory management routines.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#ifndef PME_CUDA_H
#define PME_CUDA_H

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "pme-timings.cuh"

/*
    Here is a current memory layout for the theta/dtheta B-spline float parameter arrays.
    This is the data in global memory used both by spreading and gathering kernels (with same scheduling).
    This example has PME order 4 and 2 particles per warp/data chunk.
    Each particle has 16 threads assigned to it, each thread works on 4 non-sequential global grid contributions.

    ----------------------------------------------------------------------------
    particles 0, 1                                        | particles 2, 3     | ...
    ----------------------------------------------------------------------------
    order index 0           | index 1 | index 2 | index 3 | order 0 .....
    ----------------------------------------------------------------------------
    tx0 tx1 ty0 ty1 tz0 tz1 | ..........
    ----------------------------------------------------------------------------

    Each data chunk for a single warp is 24 floats. This goes both for theta and dtheta.
    24 = 2 particles per warp *  order 4 * 3 dimensions. 48 floats (1.5 warp size) per warp in total.
    I have also tried intertwining theta and theta in a single array (they are used in pairs in gathering stage anwyay)
    and it didn't seem to make a performance difference.

    The corresponding defines follow.
 */

/* This is the distance between the neighbour theta elements - would be 2 for the intertwining layout */
#define PME_SPLINE_THETA_STRIDE 1

/* FIXME: This could be used in the code as well, but actually isn't now, only in the outdated separate spline/spread kernels */
#define PME_SPLINE_ORDER_STRIDE DIM

/* The spread/gather constant; 2 particles per warp for order of 4, depends on the templated order parameter */
#define PME_SPREADGATHER_PARTICLES_PER_WARP (warp_size / order / order)

/* FIXME: this is the shared memory size constant;
 * it depends on particlesPerBlock is another templated parameter = (BLOCK_SIZE / warp_size) * PME_SPREADGATHER_PARTICLES_PER_WARP.
 * There is a redundancy going on here.
 */
#define PME_SPREADGATHER_BLOCK_DATA_SIZE (particlesPerBlock * DIM)


// and block sizes should also be here....


/* Using textures instead of global memory. Only in spread now, but B-spline moduli in solving should also be texturized. */
#define PME_USE_TEXTURES 1

/* The internal identifiers for PME data stored on GPU and host - can be eliminated eventually */
enum PMEDataID
{
    PME_ID_THETA = 0,
    PME_ID_DTHETA,

    // grids
    PME_ID_REAL_GRID,    /* Functions as pme_grid with overlap and as fftgrid */
    PME_ID_COMPLEX_GRID, /* used only for out-of-place cuFFT, functions as cfftgrid */

    // spread (spline)
    PME_ID_FSH,
    PME_ID_NN,

    // spread
    PME_ID_XPTR,
    PME_ID_COEFFICIENT,

    // gather
    PME_ID_FORCES,

    // spread and gather
    PME_ID_IDXPTR, // grid indices as in atc->idx

    // solve
    PME_ID_ENERGY_AND_VIRIAL,
    PME_ID_BSP_MOD_XX, PME_ID_BSP_MOD_YY, PME_ID_BSP_MOD_ZZ, /* B-spline moduli */

    // end
    PME_ID_END_INVALID
};

/* The host/GPU memory tag */
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
    /* The CUDA stream where everything with PME happens */
    cudaStream_t pmeStream;

    /* Synchronization events */
    cudaEvent_t syncEnerVirD2H;    /* energy and virial have already been calculated in pme-solve, and have been copied to host */
    cudaEvent_t syncForcesD2H;     /* forces have already been calculated in pme-gather, and have been copied to host */
    cudaEvent_t syncSpreadGridD2H; /* the grid has been copied to the host after the spreading for CPU FFT */
    cudaEvent_t syncSolveGridD2H;  /* the grid has been copied to the host after the solve for CPU FFT */

    /* Permanent settings set on initialization */
    gmx_bool bGPUSolve;                                             /* Are we doing the solve stage on the GPU? Currently always TRUE */
    gmx_bool bGPUGather;                                            /* Are we doing the gather stage on the GPU? Currently always TRUE */
    gmx_bool bGPUFFT;                                               /* Are we using cuFFT as well? Currently only enabled for a single rank */
    gmx_bool bGPUSingle;                                            /* Are we using the single GPU rank? A convenience variable */
    gmx_bool bOutOfPlaceFFT;                                        /* If true, then an additional grid of the same size is used for R2C/solve/C2R */
    gmx_bool bTiming;                                               /* Enable timing using CUDA events */
    /* gmx_bool useTextureObjects; */ /* If false, then use references [unused] */

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
