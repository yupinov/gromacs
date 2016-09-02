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
 * and also names some PME CUDA memory management routines.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#ifndef PME_CUDA_H
#define PME_CUDA_H

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "pme-timings.cuh"

#include "pme-gpu.h"

/*
    Here is a current memory layout for the theta/dtheta B-spline float parameter arrays.
    This is the data in global memory used both by spreading and gathering kernels (with same scheduling).
    This example has PME order 4 and 2 particles per warp/data chunk.
    Each particle has 16 threads assigned to it, each thread works on 4 non-sequential global grid contributions.

    ----------------------------------------------------------------------------
    particles 0, 1                                        | particles 2, 3     | ...
    ----------------------------------------------------------------------------
    order index 0           | index 1 | index 2 | index 3 | order index 0 .....
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

/*! \brief \internal
 * Internal identifiers for the PME CUDA memory management (gmx_pme_cuda_t::StoragePointers).
 */
enum PMEDataID
{
    // global grids
    PME_ID_REAL_GRID = 0,     // Functions as CPU pme_grid with overlap and as fftgrid
    PME_ID_COMPLEX_GRID,      // Used only for out-of-place cuFFT, functions as CPU cfftgrid

    // spread and gather
    PME_ID_IDXPTR,            // per-particle gridline indices as in atc->idx
    PME_ID_THETA,             // B-spline values
    PME_ID_DTHETA,            // B-spline derivatives
    PME_ID_COEFFICIENT,       // charges

    // spread (spline)
    PME_ID_FSH,               // fractional shifts
    PME_ID_NN,                // gridline indices (including the neighboring cells) - basically, a modulo operation lookup table

    // spread
    PME_ID_XPTR,              // coordinates

    // gather
    PME_ID_FORCES,            // forces

    // solve
    PME_ID_ENERGY_AND_VIRIAL, // energy and virial united storage (7 floats)
    PME_ID_BSP_MOD_XX,        // B-spline moduli
    PME_ID_BSP_MOD_YY,        // B-spline moduli
    PME_ID_BSP_MOD_ZZ,        // B-spline moduli

    // end
    PME_ID_END_INVALID
};

/*! \brief \internal
 * The host/device memory tag for the PME pointer storage (gmx_pme_cuda_t::StoragePointers).
 */
enum MemLocType
{
    ML_HOST = 0,
    ML_DEVICE,
    ML_END_INVALID
};

// PME GPU structures

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
    /*! \brief
     * Reciprocal (inverted unit cell) box.
     *
     * The box is transposed as compared to the CPU pme->recipbox.
     * Basically, spread uses matrix columns (while solve and gather use rows).
     */
    float3 recipbox[DIM];
    /*! \brief Grid data dimensions - integer. */
    int3   localGridSize;
    /*! \brief Grid data dimensions - floating point. */
    float3 localGridSizeFP;
    /*! \brief Grid size dimensions - integer. The padding as compared to localGridSize includes the (order - 1) overlap. */
    int3   localGridSizePadded;
    /*! \brief Number of local atoms */
    int    nAtoms;

    /* Solving parameters - maybe they should be in a separate structure,
     * as we likely won't use GPU solve much in multi-rank PME? */

    /*! \brief The unit cell volume for solving. */
    float volume;
    /*! \brief Ewald solving coefficient = (M_PI / ewaldCoeff)^2 */
    float ewaldFactor;
    /*! \brief Electrostatics coefficient = ONE_4PI_EPS0 / pme->epsilon_r
     * This is a permanent constant.
     */
    float elFactor;
};

/*! \brief \internal
 * The main PME CUDA structure, included in the PME CPU structure by pointer.
 */
struct gmx_pme_cuda_t
{
    /*! \brief The CUDA stream where everything related to the PME happens. */
    cudaStream_t pmeStream;

    /* Synchronization events. */
    /*! \brief A synchronization event for the energy/virial being copied to the host after the solving stage. */
    cudaEvent_t syncEnerVirD2H;
    /*! \brief A synchronization event for the output forces being copied to the host after the gathering stage. */
    cudaEvent_t syncForcesD2H;
    /*! \brief A synchronization event for the grid being copied to the host after the spreading stage (for the host-side FFT). */
    cudaEvent_t syncSpreadGridD2H;
    /*! \brief A synchronization event for the grid being copied to the host after the solving stage (for the host-side FFT). */
    cudaEvent_t syncSolveGridD2H;

    /* Permanent settings set on initialization */
    /*! \brief A boolean which tells if the solving is performed on GPU. Currently always TRUE */
    gmx_bool bGPUSolve;
    /*! \brief A boolean which tells if the gathering is performed on GPU. Currently always TRUE */
    gmx_bool bGPUGather;
    /*! \brief A boolean which tells if the FFT is performed on GPU. Currently TRUE for a single MPI rank. */
    gmx_bool bGPUFFT;
    /*! \brief A convenience boolean which tells if there is only one PME GPU process. */
    gmx_bool bGPUSingle;
    /*! \brief A boolean which tells whether the complex and real grids for FFT are different or same. Currenty TRUE. */
    gmx_bool bOutOfPlaceFFT;
    /*! \brief A boolean which tells if the CUDA timing events are enabled.
     * TRUE by default, disabled by setting the environment variable GMX_DISABLE_CUDA_TIMING.
     */
    gmx_bool bTiming;
    /* gmx_bool useTextureObjects; */ /* If false, then use references [unused] */

    // constant structures for arguments
    pme_gpu_overlap_t             overlap;
    pme_gpu_const_parameters      constants;

    gmx_device_info_t            *deviceInfo;

    std::vector<pme_gpu_timing *> timingEvents;

    gmx_parallel_3dfft_gpu_t     *pfft_setup_gpu;


    /*! \brief Internal host/device pointers storage, addressed only by the PMEMemoryFetch and PMEMemoryFree routines.*/
    std::vector<void *> StoragePointers;
    /*! \brief Internal host/device pointers storage (sizes of the corresponding ranges in StoragePointers). */
    std::vector<size_t> StorageSizes;

    /* These are the host-side input/output pointers */
    /* TODO: not far in the future there will be a device input/output pointers too */
    /* Input */
    real *coordinatesHost;  /* rvec/float3 */
    real *coefficientsHost; /* real */
    /* Output (and possibly input if pme_kernel_gather does the reduction) */
    real *forcesHost;       /* rvec/float3 */
    /* Should the virial + energy live here as well? */



    /* Some device pointers/objects below (assigned from the PMEStoragePointers by PMEMemoryFetch) */

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
    real   *forces;

    float3 *coordinates;
    real   *coefficients;
};

// allocate memory; size == 0 => just fetch the current pointer
void *PMEMemoryFetch(const gmx_pme_t *pme, PMEDataID id, size_t size, MemLocType location);


// dumping all the CUDA-specific PME functions here...

// copies the bspline moduli to the device (used in PME solve)
void pme_gpu_copy_bspline_moduli(const gmx_pme_t *pme);

gmx_inline gmx_bool pme_gpu_timings_enabled(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && pme->gpu->bTiming;
}

void gmx_parallel_3dfft_destroy_gpu(const gmx_parallel_3dfft_gpu_t &pfft_setup);



// copies the nn and fsh to the device (used in PME spread(spline))
void pme_gpu_copy_calcspline_constants(const gmx_pme_t *pme);

// clearing
void pme_gpu_clear_grid(const gmx_pme_t *pme, const int grid_index);
void pme_gpu_clear_energy_virial(const gmx_pme_t *pme, const int grid_index);

// allocating
void pme_gpu_alloc_grids(const gmx_pme_t *pme, const int grid_index);
void pme_gpu_alloc_energy_virial(const gmx_pme_t *pme, const int grid_index);
void pme_gpu_realloc_gather_forces(const gmx_pme_t *pme);

void gmx_parallel_3dfft_init_gpu(gmx_parallel_3dfft_gpu_t *pfft_setup,
                                 ivec                      ndata,
                                 const gmx_pme_t          *pme);

void gmx_parallel_3dfft_complex_limits_gpu(const gmx_parallel_3dfft_gpu_t pfft_setup,
                                           ivec                           local_ndata,
                                           ivec                           local_offset,
                                           ivec                           local_size);

/*! \brief
 * Waits for the PME GPU output forces copy to the CPU buffer (pme->gpu->forcesHost) to finish.
 *
 * \param[in] pme  The PME structure.
 */
void pme_gpu_sync_output_forces(const gmx_pme_t *pme);


/*! \brief
 * Waits for the PME GPU output energy/virial copy to the CPU buffer (????) to finish.
 *
 * \param[in] pme  The PME structure.
 */
void pme_gpu_sync_energy_virial(const gmx_pme_t *pme);

#endif
