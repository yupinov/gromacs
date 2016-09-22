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

/*! \libinternal \file
 *  \brief Defines PME GPU data structure.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#ifndef PMEGPUTYPES_H
#define PMEGPUTYPES_H

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

#if GMX_GPU == GMX_GPU_CUDA
struct pme_gpu_cuda_t;
typedef pme_gpu_cuda_t pme_gpu_specific_t;
#else
typedef int pme_gpu_specific_t;
#endif

/*! \brief \internal
 * The main PME GPU structure, included in the PME CPU structure by pointer.
 */
struct gmx_pme_gpu_t
{
    /* Permanent settings set on initialization */
    /*! \brief A boolean which tells if the solving is performed on GPU. Currently always TRUE */
    gmx_bool bGPUSolve;
    /*! \brief A boolean which tells if the gathering is performed on GPU. Currently always TRUE */
    gmx_bool bGPUGather;
    /*! \brief A boolean which tells if the FFT is performed on GPU. Currently TRUE for a single MPI rank. */
    gmx_bool bGPUFFT;
    /*! \brief A convenience boolean which tells if there is only one PME GPU process. */
    gmx_bool bGPUSingle;
    /*! \brief A boolean which tells the PME to call the pme_gpu_reinit_atoms at the beginning of the run.
     * The DD pme_gpu_reinit_atoms gets called in gmx_pmeonly instead.
     * Set to TRUE initially, then to FALSE after the first MD step.
     */
    gmx_bool bNeedToUpdateAtoms;

    /*! \brief The pointer to the GPU-framework specific data. */
    pme_gpu_specific_t *mainData; /* TODO: think of a meaningful name */
};

#ifdef __cplusplus
}
#endif

#endif // PMEGPUTYPES_H
