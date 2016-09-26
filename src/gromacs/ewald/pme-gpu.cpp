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
 *  \brief Implements PME GPU functions which do not require GPU framework-specific code.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include <assert.h>
#include <string.h>
#include "gromacs/math/invertmatrix.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "pme.h"
#include "pme-grid.h"
#include "pme-solve.h"

void pme_gpu_set_io_ranges(const gmx_pme_t *pme, rvec *coordinates, rvec *forces)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    pme->gpu->forcesHost       = reinterpret_cast<float *>(forces);
    pme->gpu->coordinatesHost  = reinterpret_cast<float *>(coordinates);
    /* TODO: pin the host pointers */
}

void pme_gpu_start_step(const gmx_pme_t *pme, const matrix box)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    pme_gpu_copy_coordinates(pme);

    const size_t   boxMemorySize        = sizeof(matrix);
    const gmx_bool haveToUpdateUnitCell = memcmp(pme->gpu->previousBox, box, boxMemorySize);
    /* There could be a pressure coupling check here, but this is more straightforward.
     * This is an exact comparison of float values though.
     */
    if (haveToUpdateUnitCell)
    {
        memcpy(pme->gpu->previousBox, box, boxMemorySize);

        pme->gpu->kernelParams.step.boxVolume = box[XX][XX] * box[YY][YY] * box[ZZ][ZZ];
        assert(pme->gpu->kernelParams.step.boxVolume != 0.0f);

        matrix recipBox;
        gmx::invertBoxMatrix(box, recipBox);
        /* The GPU recipBox is transposed as compared to the CPU recipBox.
         * Spread uses matrix columns (while solve and gather use rows).
         * There is no particular reason for this; it might be further rethought/optimized for better access patterns.
         */
        const float newRecipBox[DIM][DIM] =
        {
            {recipBox[XX][XX], recipBox[YY][XX], recipBox[ZZ][XX]},
            {             0.0, recipBox[YY][YY], recipBox[ZZ][YY]},
            {             0.0,              0.0, recipBox[ZZ][ZZ]}
        };
        memcpy(pme->gpu->kernelParams.step.recipBox, newRecipBox, boxMemorySize);
    }
}

/*! \brief \internal
 * A convenience wrapper for launching either the GPU or CPU FFT.
 *
 * \param[in] pme            The PME structure.
 * \param[in] grid_index     The grid index - should currently always be 0.
 * \param[in] dir            The FFT direction enum.
 * \param[in] wcycle         The wallclock counter.
 */
void gmx_parallel_3dfft_execute_gpu_wrapper(gmx_pme_t              *pme,
                                            const int               grid_index,
                                            enum gmx_fft_direction  dir,
                                            gmx_wallcycle_t         wcycle)
{
    assert(grid_index == 0);
    if (pme_gpu_performs_FFT(pme))
    {
        wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_PME);
        pme_gpu_3dfft(pme, dir, grid_index);
        wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_PME);
    }
    else
    {
        wallcycle_start(wcycle, ewcPME_FFT);
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
        for (int thread = 0; thread < pme->nthread; thread++)
        {
            gmx_parallel_3dfft_execute(pme->pfft_setup[grid_index], dir, thread, wcycle);
        }
        wallcycle_stop(wcycle, ewcPME_FFT);
    }
}

/* Finally, the actual PME step code.
 * Together, they are a GPU counterpart to gmx_pme_do, albeit cut down due to unsupported features
 */

void pme_gpu_launch(gmx_pme_t      *pme,
                    int             nAtoms,
                    rvec            x[],
                    rvec            f[],
                    real            charges[],
                    matrix          box,
                    gmx_wallcycle_t wcycle,
                    int             flags)
{
    if (!pme_gpu_enabled(pme))
    {
        return;
    }

    pmegrids_t          *pmegrid     = NULL;
    real                *grid        = NULL;
    real                *fftgrid;
    t_complex           *cfftgrid;
    gmx_bool             bFirst;
    const gmx_bool       bCalcEnerVir            = flags & GMX_PME_CALC_ENER_VIR;
    const gmx_bool       bBackFFT                = flags & (GMX_PME_CALC_F | GMX_PME_CALC_POT);

    assert(pme->nnodes > 0);
    assert(pme->nnodes == 1 || pme->ndecompdim > 0);

    bFirst = TRUE;

    wallcycle_sub_start(wcycle, ewcsLAUNCH_GPU_PME);
    if (pme->gpu->bNeedToUpdateAtoms)
    {
        /* This only does a one-time atom data init at the first MD step.
         * Later, pme_gpu_reinit_atoms is called when needed after gmx_pme_recv_coeffs_coords.
         */
        pme_gpu_reinit_atoms(pme, nAtoms, charges);
        pme->gpu->bNeedToUpdateAtoms = FALSE;
    }
    pme_gpu_set_io_ranges(pme, x, f);              /* Should this be called every step, or on DD/DLB, or on bCalcEnerVir change? */
    pme_gpu_start_step(pme, box);                  /* This copies the coordinates, and updates the unit cell box (if it has changed) */
    wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_PME);

    const unsigned int grid_index = 0;

    /* Unpack structure */
    pmegrid     = &pme->pmegrid[grid_index];
    fftgrid     = pme->fftgrid[grid_index];
    cfftgrid    = pme->cfftgrid[grid_index];

    grid = pmegrid->grid.grid;

    // no bBackFFT, no bCalcF checks

    if (flags & GMX_PME_SPREAD)
    {
        /* Spread the coefficients on a grid */
        wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_PME);
        pme_gpu_spread(pme, &pme->atc[0], grid_index, &pmegrid->grid, bFirst, TRUE);
        wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_PME);

        //if (!pme->bUseThreads)
        {
            if (!pme_gpu_performs_wrapping(pme))
            {
                wrap_periodic_pmegrid(pme, grid);
            }

            /* sum contributions to local grid from other nodes */
#if GMX_MPI
            if (pme->nnodes > 1)
            {
                gmx_sum_qgrid_dd(pme, grid, GMX_SUM_GRID_FORWARD);
                where();
            }
#endif
            if (!pme_gpu_performs_FFT(pme))
            {
                copy_pmegrid_to_fftgrid(pme, grid, fftgrid, grid_index);
            }
        }
    }

    try
    {
        if (flags & GMX_PME_SOLVE)
        {
            /* do 3d-fft */
            gmx_parallel_3dfft_execute_gpu_wrapper(pme, grid_index, GMX_FFT_REAL_TO_COMPLEX,
                                                   wcycle);

            /* solve in k-space for our local cells */
            if (pme_gpu_performs_solve(pme))
            {
                wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_PME);
                pme_gpu_solve(pme, cfftgrid, bCalcEnerVir);
                wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_PME);
            }
            else
            {
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
                for (int thread = 0; thread < pme->nthread; thread++)
                {
                    solve_pme_yzx(pme, cfftgrid,
                                  box[XX][XX]*box[YY][YY]*box[ZZ][ZZ],
                                  bCalcEnerVir, pme->nthread, thread);
                }
            }
        }

        if (bBackFFT)
        {
            /* do 3d-invfft */
            gmx_parallel_3dfft_execute_gpu_wrapper(pme, grid_index, GMX_FFT_COMPLEX_TO_REAL, wcycle);

            if (!pme_gpu_performs_FFT(pme) || !pme_gpu_performs_gather(pme))
            {
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
                for (int thread = 0; thread < pme->nthread; thread++)
                {
                    copy_fftgrid_to_pmegrid(pme, fftgrid, grid, grid_index, pme->nthread, thread);
                }
            }
        }
    } GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;

    if (bBackFFT)
    {
        /* distribute local grid to all nodes */
        if (!pme_gpu_performs_wrapping(pme))
        {
            unwrap_periodic_pmegrid(pme, grid);
        }
    }
}

// this will only copy the forces buffer (with results from listed calculations, etc.) to the GPU (for bClearF == false),
// launch the gather kernel, copy the result back
void pme_gpu_launch_gather(const gmx_pme_t                 *pme,
                           gmx_wallcycle_t gmx_unused       wcycle,
                           gmx_bool                         bClearForces)
{
    if (!pme_gpu_performs_gather(pme))
    {
        return;
    }

    wallcycle_sub_start_nocount(wcycle, ewcsLAUNCH_GPU_PME);
    pme_gpu_gather(pme, bClearForces);
    wallcycle_sub_stop(wcycle, ewcsLAUNCH_GPU_PME);
}

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
