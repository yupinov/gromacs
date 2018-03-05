/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013-2016,2017,2018, by the GROMACS development team, led by
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
 *  \brief Implements PME GPU spline calculation and charge spreading in CUDA.
 *  TODO: consider always pre-sorting particles (as in DD case).
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include <cassert>

#include "gromacs/gpu_utils/gputraits_ocl.h"

#include "gromacs/ewald/pme.h"
//#include "gromacs/gpu_utils/cuda_kernel_utils.cuh"
//#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"

#include "pme-types-ocl.h"
#include "pme-grid.h"
//#include "pme-timings.cuh"

void pme_gpu_spread(PmeGpu    *pmeGpu,
                    int gmx_unused   gridIndex,
                    real            *h_grid,
                    bool             computeSplines,
                    bool             spreadCharges)
{
    GMX_ASSERT(computeSplines || spreadCharges, "PME spline/spread kernel has invalid input (nothing to do)");
    CommandStream stream          = pmeGpu->archSpecific->pmeStream;
    const auto   *kernelParamsPtr = pmeGpu->kernelParams.get();
    GMX_ASSERT(kernelParamsPtr->atoms.nAtoms > 0, "No atom data in PME GPU spread");

    // FIXME this has to be computed during init? at keast getwarpSize
    //! Spreading max block size in threads
    constexpr int c_spreadMaxThreadsPerBlock = c_spreadMaxWarpsPerBlock * warp_size;

    const int order         = pmeGpu->common->pme_order;
    const int atomsPerBlock = c_spreadMaxThreadsPerBlock / PME_SPREADGATHER_THREADS_PER_ATOM;
    // TODO: pick smaller block size in runtime if needed
    // (e.g. on 660 Ti where 50% occupancy is ~25% faster than 100% occupancy with RNAse (~17.8k atoms))
    // If doing so, change atomsPerBlock in the kernels as well.
    // TODO: test varying block sizes on modern arch-s as well
    // TODO: also consider using cudaFuncSetCacheConfig() for preferring shared memory on older architectures
    //(for spline data mostly, together with varying PME_GPU_PARALLEL_SPLINE define)
    GMX_ASSERT(!c_usePadding || !(PME_ATOM_DATA_ALIGNMENT % atomsPerBlock), "inconsistent atom data padding vs. spreading block size");

    const int blockCount = pmeGpu->nAtomsPadded / atomsPerBlock;
    //FIXME pickup Fermi fix pmeGpuCreateGrid(pmeGpu, blockCount);

    KernelLaunchConfig config;
    config.sharedMemorySize = 0;
    config.stream      = stream;

    config.blockSize.x = order;
    config.blockSize.y = order;
    config.blockSize.z = atomsPerBlock;
    config.gridSize.x  = blockCount;
    config.gridSize.y  = 1;
    config.gridSize.z  = 1;

    // FIXME unite dim3/size_t[3]
    //FIXME mvoe to launcher*
    /*
    config.gridSize.x *= config.blockSize.x;
    config.gridSize.y *= config.blockSize.y;
    config.gridSize.z *= config.blockSize.z;
    */

    
    // These should later check for PME decomposition
    const bool wrapX = true;
    const bool wrapY = true;
    GMX_UNUSED_VALUE(wrapX);
    GMX_UNUSED_VALUE(wrapY);
    switch (order)
    {
        case 4: // FIXME make thsi conditionals go away
        {
            // TODO: cleaner unroll with some template trick?
            if (computeSplines)
            {
                if (spreadCharges)
                {
                    pme_gpu_start_timing(pmeGpu, gtPME_SPLINEANDSPREAD);
                    //launchGpuKernel(config, pme_spline_and_spread_kernel<4, true, true, wrapX, wrapY>, kernelParamsPtr);
                    launchGpuKernel(config, pmeGpu->archSpecific->splineAndSpreadKernel, kernelParamsPtr);
                    //FIXME error handling? CU_LAUNCH_ERR("pme_spline_and_spread_kernel"); - common result?
                    pme_gpu_stop_timing(pmeGpu, gtPME_SPLINEANDSPREAD);
                }
                else
                {
                    pme_gpu_start_timing(pmeGpu, gtPME_SPLINE);
                    //launchGpuKernel(config, pme_spline_and_spread_kernel<4, true, false, wrapX, wrapY>, kernelParamsPtr);
                    launchGpuKernel(config, pmeGpu->archSpecific->splineKernel, kernelParamsPtr);
                    //CU_LAUNCH_ERR("pme_spline_and_spread_kernel");
                    pme_gpu_stop_timing(pmeGpu, gtPME_SPLINE);
                }
            }
            else
            {
                pme_gpu_start_timing(pmeGpu, gtPME_SPREAD);
                launchGpuKernel(config, pmeGpu->archSpecific->spreadKernel, kernelParamsPtr); // TODO make those namespointers in CUDA version!
                //launchGpuKernel(config, pme_spline_and_spread_kernel<4, false, true, wrapX, wrapY>, kernelParamsPtr);
                //CU_LAUNCH_ERR("pme_spline_and_spread_kernel");
                pme_gpu_stop_timing(pmeGpu, gtPME_SPREAD);
            }
        }
        break;

        default:
            GMX_THROW(gmx::NotImplementedError("The code for pme_order != 4 was not tested!"));
    }

    const bool copyBackGrid = spreadCharges && (pme_gpu_is_testing(pmeGpu) || !pme_gpu_performs_FFT(pmeGpu));
    if (copyBackGrid)
    {
        pme_gpu_copy_output_spread_grid(pmeGpu, h_grid);
    }
    const bool copyBackAtomData = computeSplines && (pme_gpu_is_testing(pmeGpu) || !pme_gpu_performs_gather(pmeGpu));
    if (copyBackAtomData)
    {
        pme_gpu_copy_output_spread_atom_data(pmeGpu);
    }
}
