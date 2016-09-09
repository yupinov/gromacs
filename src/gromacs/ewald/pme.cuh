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


/* Using textures instead of global memory. Only in spread now, but B-spline moduli in solving could also be texturized. */
#define PME_USE_TEXTURES 1
#if PME_USE_TEXTURES
/* Using texture objects as opposed to texture references
 * FIXME: rely entirely on dynamic device info instead, remove more ugly #ifs
 */
#define PME_USE_TEXOBJ 1
#endif


//yupinov fractional shifts
// gridline indices (including the neighboring cells) - basically, a modulo operation lookup table
// Functions as CPU pme_grid with overlap and as fftgrid
// Used only for out-of-place cuFFT, functions as CPU cfftgrid

/* PME GPU data structures.
 * They describe all the fixed-size data that needs to be accesses by the CUDA kernels.
 * Pointers to the particle/grid/spline data live here as well.
 * The data is split into sub-structures depending on its update rate.
 */

/*! \brief \internal
 * A GPU data structure for storing the constant PME data.
 *
 * This only has to be initialized once.
 */
struct pme_gpu_const_params
{
    /*! \brief Electrostatics coefficient = ONE_4PI_EPS0 / pme->epsilon_r */
    float elFactor;
};

/*! \brief \internal
 * A GPU data structure for storing the PME data related to the grid size and cut-off.
 *
 * This only has to be updated every DLB step.
 */
struct pme_gpu_grid_params
{
    /* Grid sizes */
    /*! \brief Grid data dimensions - integer. */
    int3   localGridSize;
    /*! \brief Grid data dimensions - floating point. */
    float3 localGridSizeFP;
    /*! \brief Grid size dimensions - integer. The padding as compared to localGridSize includes the (order - 1) overlap. */
    int3   localGridSizePadded;

    /* Grid pointers */
    /*! \brief Real space grid. */
    float *realGrid;
    /*! \brief Complex grid - used in FFT/solve.
     *
     * If we're using inplace cuFFT, then it's the same pointer as realGrid.
     */
    float2 *fourierGrid;

    /* Crude wrap/unwrap overlap zone sizes - can go away with a better rewrite of wrap/unwrap */
#define OVERLAP_ZONES 7
    /*! \brief The Y and Z sizes of the overlap zones */
    int2 overlapSizes[OVERLAP_ZONES];
    /*! \brief The total cell counts of the overlap zones */
    int  overlapCellCounts[OVERLAP_ZONES];

    /*! \brief Ewald solving factor = (M_PI / ewaldCoeff)^2 */
    float ewaldFactor;

    /*! \brief Fractional shifts as in pme->fshx/fshy/fshz, laid out sequentially (XXX....XYYY......YZZZ.....Z) */
    float               *fshArray;
    /*! \brief Fractional shifts - a texture object for accessing fshArray */
    cudaTextureObject_t  fshTexture;
    /*! \brief Fractional shifts gridline indices
     * (modulo lookup table as in pme->nnx/nny/nnz, laid out sequentially (XXX....XYYY......YZZZ.....Z))
     */
    int                *nnArray;
    /*! \brief Fractional shifts gridline indices - a texture object for accessing nnArray */
    cudaTextureObject_t nnTexture;
    /*! \brief Offsets for X/Y/Z components of fshArray and nnArray */
    int3                fshOffset;

    /*! \brief Grid spline values as in pme->bsp_mod
     * (laid out sequentially (XXX....XYYY......YZZZ.....Z))
     */
    float              *splineValuesArray;
    /*! \brief Offsets for X/Y/Z components of splineValuesArray */
    int3                splineValuesOffset;

};

/*! \brief \internal
 * A GPU data structure for storing the PME data of the atoms, local to this process' domain partition.
 *
 * This only has to be updated every DD step.
 */
struct pme_gpu_atom_params
{
    /*! \brief Number of local atoms */
    int    nAtoms;
    /*! \brief Pointer to the global GPU memory with input rvec atom coordinates.
     * The coordinates themselves change and need to be copied to the GPU every MD step,
     * but the pointer changes only on DD.
     */
    float3 *coordinates;
    /*! \brief Pointer to the global GPU memory with input atom charges.
     * The charges only need to be reallocated and copied to the GPU on DD step.
     */
    float  *coefficients;
    /*! \brief Pointer to the global GPU memory with input/output rvec atom forces.
     * The forces change and need to be copied from (and possibly to) the GPU every MD step,
     * but the pointer changes only on DD.
     */
    float  *forces;
    /*! \brief Pointer to the global GPU memory with ivec atom gridline indices.
     * Computed on GPU in the spline calculation part.
     */
    int *gridlineIndices;

    /* B-spline parameters are computed entirely on GPU every MD step, not copied.
     * Unless we want to try something like GPU spread + CPU gather?
     */
    /*! \brief Pointer to the global GPU memory with B-spline values */
    float  *theta;
    /*! \brief Pointer to the global GPU memory with B-spline derivative values */
    float  *dtheta;
};

/*! \brief \internal
 * A GPU data structure for storing the PME data which might change every MD step.
 */
struct pme_gpu_step_params
{
    /* The box parameters. The box only changes size each step with pressure coupling enabled.
     * How about a corresponding check in the code? */
    /*! \brief
     * Reciprocal (inverted unit cell) box.
     *
     * The box is transposed as compared to the CPU pme->recipbox.
     * Basically, spread uses matrix columns (while solve and gather use rows).
     * This storage format might be not the best since the box is always triangular.
     */
    float3 recipbox[DIM];
    /*! \brief The unit cell volume for solving. */
    float  boxVolume;
};

/*! \brief \internal
 * A single structure encompassing all the PME data used on GPU.
 */
struct pme_gpu_kernel_params
{
    /*! \brief Constant data. */
    pme_gpu_const_params constants;
    /*! \brief Data dependent on the grid size/cutoff. */
    pme_gpu_grid_params  grid;
    /*! \brief Data dependent on the DD and local atoms. */
    pme_gpu_atom_params  atoms;
    /*! \brief Data that changes on every MD step. */
    pme_gpu_step_params  step;
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

    //gmx_bool bUseTextureObjects;  /* If false, then use references [unused] */

    /*! \brief A single structure encompassing all the PME data used on GPU.
     * This is the only parameter to all the PME CUDA kernels.
     * Can probably be copied to the constant GPU memory once per MD step instead of being a parameter.
     */
    pme_gpu_kernel_params                kernelParams;

    gmx_device_info_t                   *deviceInfo;

    pme_gpu_timing              *        timingEvents[gtPME_EVENT_COUNT];

    gmx_parallel_3dfft_gpu_t            *pfft_setup_gpu;

    /* These are the host-side input/output pointers */
    /* TODO: not far in the future there will be a device input/output pointers too */
    /* Input */
    float  *coordinatesHost;  /* rvec/float3 */
    float  *coefficientsHost;
    /* Output (and possibly input if pme_kernel_gather does the reduction) */
    float  *forcesHost;      /* rvec/float3 */
    /* Should the virial + energy live here as well? */
    /*! \brief Energy and virial intermediate host-side buffer, managed and pinned by PME GPU entirely. Size is 7 floats. */
    float *energyAndVirialHost;
    /*! \brief B-spline values (temporary?) intermediate host-side buffers, managed and pinned by PME GPU entirely. Sizes are the grid sizes. */
    float *splineValuesHost[DIM];
    /*! \brief Sizes of the corresponding splineValuesHost arrays in bytes */
    size_t splineValuesHostSizes[DIM]; //oh god the names

    /* Some device pointers/objects below (assigned from the PMEStoragePointers by PMEMemoryFetch) */

    // solve
    // 6 virial components, energy => 7 elements
    float  *energyAndVirial;
    size_t  energyAndVirialSizeBytes; // (7 * sizeof(float))

    /* GPU allocation sizes (counts, not bytes!) */
    int coordinatesSize;
    int coordinatesSizeAlloc;
    int forcesSize;
    int forcesSizeAlloc;
    int gridlineIndicesSize;
    int gridlineIndicesSizeAlloc;
    int splineDataSize;
    int splineDataSizeAlloc;
    int coefficientsSize;
    int coefficientsSizeAlloc;
    int fractShiftsSize;
    int fractShiftsSizeAlloc;
    int gridSize;
    int gridSizeAlloc;
    int splineValuesSize;
    int splineValuesSizeAlloc;
};

// dumping all the CUDA-specific PME functions here...

gmx_inline gmx_bool pme_gpu_timings_enabled(const gmx_pme_t *pme)
{
    return pme_gpu_enabled(pme) && pme->gpu->bTiming;
}

void gmx_parallel_3dfft_destroy_gpu(const gmx_parallel_3dfft_gpu_t &pfft_setup);



// copies the nn and fsh to the device (used in PME spread(spline))
void pme_gpu_realloc_and_copy_fract_shifts(const gmx_pme_t *pme);
void pme_gpu_free_fract_shifts(const gmx_pme_t *pme);

// clearing
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