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
 *  \brief Implements PME GPU functions which do not require framework-specific code.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include "gromacs/utility/fatalerror.h"
#include "pme.h"

void pme_gpu_get_results(const gmx_pme_t *pme,
                         gmx_wallcycle_t  wcycle,
                         matrix           vir_q,
                         real            *energy_q,
                         int              flags)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    const gmx_bool       bCalcEnerVir            = flags & GMX_PME_CALC_ENER_VIR;
    const gmx_bool       bCalcF                  = flags & GMX_PME_CALC_F;

    wallcycle_sub_start(wcycle, ewcsWAIT_GPU_PME);
    pme_gpu_finish_step(pme, bCalcF, bCalcEnerVir);
    wallcycle_sub_stop(wcycle, ewcsWAIT_GPU_PME);

    if (bCalcEnerVir)
    {
        if (pme->doCoulomb)
        {
            pme_gpu_get_energy_virial(pme, energy_q, vir_q);
            if (debug)
            {
                fprintf(debug, "Electrostatic PME mesh energy [GPU]: %g\n", *energy_q);
            }
        }
        else
        {
            *energy_q = 0;
        }
    }
    /* No bCalcF code since currently forces are copied to the output host buffer with no transformation. */
}
