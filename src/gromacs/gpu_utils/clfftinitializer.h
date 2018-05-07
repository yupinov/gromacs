/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018, by the GROMACS development team, led by
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
 * \brief
 * Declares ClfftInitializer class, which initializes and
 * tears down the clFFT library resources in OpenCL builds,
 * and does nothing in other builds.
 * clFFT itself is used in the OpenCL implementation of PME
 * for 3D R2C/C2R transforms. It is know to work with NVidia
 * OpenCL, AMD fglrx and AMDGPU-PRO drivers, and to not work with
 * AMD Rocm dev preview as of May 2018 (#2515).
 * TODO: find out compatibility with Intel once the rest of PME
 * gets there (#2516), or by building clFFT own tests.
 *
 * \inlibraryapi
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 */
#ifndef GMX_GPU_UTILS_CLFFTINITIALIZER_H
#define GMX_GPU_UTILS_CLFFTINITIALIZER_H

#include "gromacs/utility/classhelpers.h"

namespace gmx
{

class ClfftInitializer
{
    public:
        ClfftInitializer();
        ~ClfftInitializer();

        GMX_DISALLOW_COPY_AND_ASSIGN(ClfftInitializer);
};

}

#endif
