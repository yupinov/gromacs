/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2014,2015, by the GROMACS development team, led by
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
#ifndef GMX_EWALD_PME_SOLVE_H
#define GMX_EWALD_PME_SOLVE_H

#include "gromacs/math/gmxcomplex.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

#include "pme-gpu.h"

struct pme_solve_work_t
{
    /* work data for solve_pme */
    int      nalloc;
    real *   mhx;
    real *   mhy;
    real *   mhz;
    real *   m2;
    real *   denom;
    real *   tmp1_alloc;
    real *   tmp1;
    real *   tmp2;
    real *   eterm;
    real *   m2inv;

    real     energy_q;
    matrix   vir_q;
    real     energy_lj;
    matrix   vir_lj;
};

struct gmx_pme_t;

/*! \brief Allocates array of work structures
 *
 * Note that work is the address of a pointer allocated by
 * this function. Upon return it will point at
 * an array of work structures.
 */
void pme_init_all_work(struct pme_solve_work_t **work, int nthread, int nkx);

/*! \brief Frees array of work structures
 *
 * Frees work and sets it to NULL. */
void pme_free_all_work(struct pme_solve_work_t **work, int nthread);

/*! \brief Get energy and virial for electrostatics
 *
 * Note that work is an array of work structures
 */
void get_pme_ener_vir_q(struct pme_solve_work_t *work, int nthread,
                        real *mesh_energy, matrix vir);

/*! \brief Get energy and virial for L-J
 *
 * Note that work is an array of work structures
 */
void get_pme_ener_vir_lj(struct pme_solve_work_t *work, int nthread,
                         real *mesh_energy, matrix vir);

int solve_pme_yzx(struct gmx_pme_t *pme, t_complex *grid,
                  real ewaldcoeff, real vol,
                  gmx_bool bEnerVir,
                  int nthread, int thread);

int solve_pme_lj_yzx(struct gmx_pme_t *pme, t_complex **grid, gmx_bool bLB,
                     real ewaldcoeff, real vol,
                     gmx_bool bEnerVir, int nthread, int thread);

#include "pme-internal.h"
//yupinov lj solve
//yupinov move GPU stuff?

inline int solve_pme_lj_yzx_wrapper(struct gmx_pme_t *pme, t_complex **grid, gmx_bool bLB,
                     real ewaldcoeff, real vol,
                     gmx_bool bEnerVir, int nthread, int thread)
{
    int res = -1;
    /*
     * if (pme->bGPU)
        ;//yupinov
    else
    */
        res = solve_pme_lj_yzx(pme, grid, bLB,ewaldcoeff, vol, bEnerVir, nthread, thread);
    return res;
}

inline int solve_pme_yzx_wrapper(struct gmx_pme_t *pme, t_complex *grid,
                  real ewaldcoeff, real vol,
                  gmx_bool bEnerVir,
                  int nthread, int thread)
{
    int res = 0;
    if (pme->bGPU)
        solve_pme_gpu(pme, grid, ewaldcoeff, vol, bEnerVir);
    else
        res = solve_pme_yzx(pme, grid, ewaldcoeff, vol, bEnerVir, nthread, thread);
    return res;
}

#endif
