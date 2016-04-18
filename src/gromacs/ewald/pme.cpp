/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016, by the GROMACS development team, led by
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
 *
 * \brief This file contains function definitions necessary for
 * computing energies and forces for the PME long-ranged part (Coulomb
 * and LJ).
 *
 * \author Erik Lindahl <erik@kth.se>
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_ewald
 */
/* IMPORTANT FOR DEVELOPERS:
 *
 * Triclinic pme stuff isn't entirely trivial, and we've experienced
 * some bugs during development (many of them due to me). To avoid
 * this in the future, please check the following things if you make
 * changes in this file:
 *
 * 1. You should obtain identical (at least to the PME precision)
 *    energies, forces, and virial for
 *    a rectangular box and a triclinic one where the z (or y) axis is
 *    tilted a whole box side. For instance you could use these boxes:
 *
 *    rectangular       triclinic
 *     2  0  0           2  0  0
 *     0  2  0           0  2  0
 *     0  0  6           2  2  6
 *
 * 2. You should check the energy conservation in a triclinic box.
 *
 * It might seem an overkill, but better safe than sorry.
 * /Erik 001109
 */
//yupinov ^

#include "gmxpre.h"

#include "pme.h"

#include "config.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "gromacs/fft/parallel_3dfft.h"
#include "gromacs/fileio/pdbio.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/math/gmxcomplex.h"
#include "gromacs/math/invertmatrix.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/timing/cyclecounter.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/gmxomp.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"

#include "calculate-spline-moduli.h"
#include "pme-gather.h"
#include "pme-gpu.h"
#include "pme-grid.h"
#include "pme-internal.h"
#include "pme-redistribute.h"
#include "pme-solve.h"
#include "pme-spline-work.h"
#include "pme-spread.h"

/*! \brief Number of bytes in a cache line.
 *
 * Must also be a multiple of the SIMD and SIMD4 register size, to
 * preserve alignment.
 */
const int gmxCacheLineSize = 64;

//! Set up coordinate communication
static void setup_coordinate_communication(pme_atomcomm_t *atc)
{
    int nslab, n, i;
    int fw, bw;

    nslab = atc->nslab;

    n = 0;
    for (i = 1; i <= nslab/2; i++)
    {
        fw = (atc->nodeid + i) % nslab;
        bw = (atc->nodeid - i + nslab) % nslab;
        if (n < nslab - 1)
        {
            atc->node_dest[n] = fw;
            atc->node_src[n]  = bw;
            n++;
        }
        if (n < nslab - 1)
        {
            atc->node_dest[n] = bw;
            atc->node_src[n]  = fw;
            n++;
        }
    }
}

int gmx_pme_destroy(struct gmx_pme_t **pmedata)
{
    int i;

    sfree((*pmedata)->nnx);
    sfree((*pmedata)->nny);
    sfree((*pmedata)->nnz);

    for (i = 0; i < (*pmedata)->ngrids; ++i)
    {
        pmegrids_destroy(&(*pmedata)->pmegrid[i]);
        gmx_parallel_3dfft_destroy((*pmedata)->pfft_setup[i]);
    }

    sfree((*pmedata)->lb_buf1);
    sfree((*pmedata)->lb_buf2);

    pme_free_all_work(&(*pmedata)->solve_work, (*pmedata)->nthread);

    pme_gpu_deinit(&(*pmedata));

    sfree(*pmedata);
    *pmedata = NULL;

    return 0;
}

/*! \brief Round \p n up to the next multiple of \p f */
static int mult_up(int n, int f)
{
    return ((n + f - 1)/f)*f;
}

/*! \brief Return estimate of the load imbalance from the PME grid not being a good match for the number of PME ranks */
static double estimate_pme_load_imbalance(struct gmx_pme_t *pme)
{
    int    nma, nmi;
    double n1, n2, n3;

    nma = pme->nnodes_major;
    nmi = pme->nnodes_minor;

    n1 = mult_up(pme->nkx, nma)*mult_up(pme->nky, nmi)*pme->nkz;
    n2 = mult_up(pme->nkx, nma)*mult_up(pme->nkz, nmi)*pme->nky;
    n3 = mult_up(pme->nky, nma)*mult_up(pme->nkz, nmi)*pme->nkx;

    /* pme_solve is roughly double the cost of an fft */

    return (n1 + n2 + 3*n3)/(double)(6*pme->nkx*pme->nky*pme->nkz);
}

/*! \brief Initialize atom communication data structure */
static void init_atomcomm(struct gmx_pme_t *pme, pme_atomcomm_t *atc,
                          int dimind, gmx_bool bSpread)
{
    int thread;

    atc->dimind    = dimind;
    atc->nslab     = 1;
    atc->nodeid    = 0;
    atc->pd_nalloc = 0;
#if GMX_MPI
    if (pme->nnodes > 1)
    {
        atc->mpi_comm = pme->mpi_comm_d[dimind];
        MPI_Comm_size(atc->mpi_comm, &atc->nslab);
        MPI_Comm_rank(atc->mpi_comm, &atc->nodeid);
    }
    if (debug)
    {
        fprintf(debug, "For PME atom communication in dimind %d: nslab %d rank %d\n", atc->dimind, atc->nslab, atc->nodeid);
    }
#endif

    atc->bSpread   = bSpread;
    atc->pme_order = pme->pme_order;

    if (atc->nslab > 1)
    {
        snew(atc->node_dest, atc->nslab);
        snew(atc->node_src, atc->nslab);
        setup_coordinate_communication(atc);

        snew(atc->count_thread, pme->nthread);
        for (thread = 0; thread < pme->nthread; thread++)
        {
            snew(atc->count_thread[thread], atc->nslab);
        }
        atc->count = atc->count_thread[0];
        snew(atc->rcount, atc->nslab);
        snew(atc->buf_index, atc->nslab);
    }

    atc->nthread = pme->nthread;
    if (atc->nthread > 1)
    {
        snew(atc->thread_plist, atc->nthread);
    }
    snew(atc->spline, atc->nthread);
    for (thread = 0; thread < atc->nthread; thread++)
    {
        if (atc->nthread > 1)
        {
            snew(atc->thread_plist[thread].n, atc->nthread+2*gmxCacheLineSize);
            atc->thread_plist[thread].n += gmxCacheLineSize;
        }
        snew(atc->spline[thread].thread_one, pme->nthread);
        atc->spline[thread].thread_one[thread] = 1;
    }
}

/*! \brief Initialize data structure for communication */
static void
init_overlap_comm(pme_overlap_t *  ol,
                  int              norder,
#if GMX_MPI
                  MPI_Comm         comm,
#endif
                  int              nnodes,
                  int              nodeid,
                  int              ndata,
                  int              commplainsize)
{
    int              b, i;
    pme_grid_comm_t *pgc;
    gmx_bool         bCont;
    int              fft_start, fft_end, send_index1, recv_index1;
#if GMX_MPI
    MPI_Status       stat;

    ol->mpi_comm = comm;
#endif

    ol->nnodes = nnodes;
    ol->nodeid = nodeid;

    /* Linear translation of the PME grid won't affect reciprocal space
     * calculations, so to optimize we only interpolate "upwards",
     * which also means we only have to consider overlap in one direction.
     * I.e., particles on this node might also be spread to grid indices
     * that belong to higher nodes (modulo nnodes)
     */

    snew(ol->s2g0, ol->nnodes+1);
    snew(ol->s2g1, ol->nnodes);
    if (debug)
    {
        fprintf(debug, "PME slab boundaries:");
    }
    for (i = 0; i < nnodes; i++)
    {
        /* s2g0 the local interpolation grid start.
         * s2g1 the local interpolation grid end.
         * Since in calc_pidx we divide particles, and not grid lines,
         * spatially uniform along dimension x or y, we need to round
         * s2g0 down and s2g1 up.
         */
        ol->s2g0[i] = ( i   *ndata + 0       )/nnodes;
        ol->s2g1[i] = ((i+1)*ndata + nnodes-1)/nnodes + norder - 1;

        if (debug)
        {
            fprintf(debug, "  %3d %3d", ol->s2g0[i], ol->s2g1[i]);
        }
    }
    ol->s2g0[nnodes] = ndata;
    if (debug)
    {
        fprintf(debug, "\n");
    }

    /* Determine with how many nodes we need to communicate the grid overlap */
    b = 0;
    do
    {
        b++;
        bCont = FALSE;
        for (i = 0; i < nnodes; i++)
        {
            if ((i+b <  nnodes && ol->s2g1[i] > ol->s2g0[i+b]) ||
                (i+b >= nnodes && ol->s2g1[i] > ol->s2g0[i+b-nnodes] + ndata))
            {
                bCont = TRUE;
            }
        }
    }
    while (bCont && b < nnodes);
    ol->noverlap_nodes = b - 1;

    snew(ol->send_id, ol->noverlap_nodes);
    snew(ol->recv_id, ol->noverlap_nodes);
    for (b = 0; b < ol->noverlap_nodes; b++)
    {
        ol->send_id[b] = (ol->nodeid + (b + 1)) % ol->nnodes;
        ol->recv_id[b] = (ol->nodeid - (b + 1) + ol->nnodes) % ol->nnodes;
    }
    snew(ol->comm_data, ol->noverlap_nodes);

    ol->send_size = 0;
    for (b = 0; b < ol->noverlap_nodes; b++)
    {
        pgc = &ol->comm_data[b];
        /* Send */
        fft_start        = ol->s2g0[ol->send_id[b]];
        fft_end          = ol->s2g0[ol->send_id[b]+1];
        if (ol->send_id[b] < nodeid)
        {
            fft_start += ndata;
            fft_end   += ndata;
        }
        send_index1       = ol->s2g1[nodeid];
        send_index1       = std::min(send_index1, fft_end);
        pgc->send_index0  = fft_start;
        pgc->send_nindex  = std::max(0, send_index1 - pgc->send_index0);
        ol->send_size    += pgc->send_nindex;

        /* We always start receiving to the first index of our slab */
        fft_start        = ol->s2g0[ol->nodeid];
        fft_end          = ol->s2g0[ol->nodeid+1];
        recv_index1      = ol->s2g1[ol->recv_id[b]];
        if (ol->recv_id[b] > nodeid)
        {
            recv_index1 -= ndata;
        }
        recv_index1      = std::min(recv_index1, fft_end);
        pgc->recv_index0 = fft_start;
        pgc->recv_nindex = std::max(0, recv_index1 - pgc->recv_index0);
    }

#if GMX_MPI
    /* Communicate the buffer sizes to receive */
    for (b = 0; b < ol->noverlap_nodes; b++)
    {
        MPI_Sendrecv(&ol->send_size, 1, MPI_INT, ol->send_id[b], b,
                     &ol->comm_data[b].recv_size, 1, MPI_INT, ol->recv_id[b], b,
                     ol->mpi_comm, &stat);
    }
#endif

    /* For non-divisible grid we need pme_order iso pme_order-1 */
    snew(ol->sendbuf, norder*commplainsize);
    snew(ol->recvbuf, norder*commplainsize);
}

void gmx_pme_check_restrictions(int pme_order,
                                int nkx, int nky, int nkz,
                                int nnodes_major,
                                int nnodes_minor,
                                gmx_bool bUseThreads,
                                gmx_bool bFatal,
                                gmx_bool *bValidSettings)
{
    if (pme_order > PME_ORDER_MAX)
    {
        if (!bFatal)
        {
            *bValidSettings = FALSE;
            return;
        }
        gmx_fatal(FARGS, "pme_order (%d) is larger than the maximum allowed value (%d). Modify and recompile the code if you really need such a high order.",
                  pme_order, PME_ORDER_MAX);
    }

    if (nkx <= pme_order*(nnodes_major > 1 ? 2 : 1) ||
        nky <= pme_order*(nnodes_minor > 1 ? 2 : 1) ||
        nkz <= pme_order)
    {
        if (!bFatal)
        {
            *bValidSettings = FALSE;
            return;
        }
        gmx_fatal(FARGS, "The PME grid sizes need to be larger than pme_order (%d) and for dimensions with domain decomposition larger than 2*pme_order",
                  pme_order);
    }

    /* Check for a limitation of the (current) sum_fftgrid_dd code.
     * We only allow multiple communication pulses in dim 1, not in dim 0.
     */
    if (bUseThreads && (nkx < nnodes_major*pme_order &&
                        nkx != nnodes_major*(pme_order - 1)))
    {
        if (!bFatal)
        {
            *bValidSettings = FALSE;
            return;
        }
        gmx_fatal(FARGS, "The number of PME grid lines per rank along x is %g. But when using OpenMP threads, the number of grid lines per rank along x should be >= pme_order (%d) or = pmeorder-1. To resolve this issue, use fewer ranks along x (and possibly more along y and/or z) by specifying -dd manually.",
                  nkx/(double)nnodes_major, pme_order);
    }

    if (bValidSettings != NULL)
    {
        *bValidSettings = TRUE;
    }

    return;
}

/*! \brief Round \p enumerator */
static int div_round_up(int enumerator, int denominator)
{
    return (enumerator + denominator - 1)/denominator;
}

int gmx_pme_init(struct gmx_pme_t **pmedata,
                 t_commrec *        cr,
                 int                nnodes_major,
                 int                nnodes_minor,
                 t_inputrec *       ir,
                 int                homenr,
                 gmx_bool           bFreeEnergy_q,
                 gmx_bool           bFreeEnergy_lj,
                 gmx_bool           bReproducible,
                 int                nthread,
                 gmx_bool           bPMEGPU,
                 const gmx_hw_info_t *hwinfo,
                 const gmx_gpu_opt_t *gpu_opt)
{
    struct gmx_pme_t *pme = NULL;

    int               use_threads, sum_use_threads, i;
    ivec              ndata;

    if (debug)
    {
        fprintf(debug, "Creating PME data structures.\n");
    }
    snew(pme, 1);

    pme->sum_qgrid_tmp       = NULL;
    pme->sum_qgrid_dd_tmp    = NULL;
    pme->buf_nalloc          = 0;

    pme->nnodes              = 1;
    pme->bPPnode             = TRUE;

    pme->nnodes_major        = nnodes_major;
    pme->nnodes_minor        = nnodes_minor;

#if GMX_MPI
    if (nnodes_major*nnodes_minor > 1)
    {
        pme->mpi_comm = cr->mpi_comm_mygroup;

        MPI_Comm_rank(pme->mpi_comm, &pme->nodeid);
        MPI_Comm_size(pme->mpi_comm, &pme->nnodes);
        if (pme->nnodes != nnodes_major*nnodes_minor)
        {
            gmx_incons("PME rank count mismatch");
        }
    }
    else
    {
        pme->mpi_comm = MPI_COMM_NULL;
    }
#endif

    if (pme->nnodes == 1)
    {
#if GMX_MPI
        pme->mpi_comm_d[0] = MPI_COMM_NULL;
        pme->mpi_comm_d[1] = MPI_COMM_NULL;
#endif
        pme->ndecompdim   = 0;
        pme->nodeid_major = 0;
        pme->nodeid_minor = 0;
#if GMX_MPI
        pme->mpi_comm_d[0] = pme->mpi_comm_d[1] = MPI_COMM_NULL;
#endif
    }
    else
    {
        if (nnodes_minor == 1)
        {
#if GMX_MPI
            pme->mpi_comm_d[0] = pme->mpi_comm;
            pme->mpi_comm_d[1] = MPI_COMM_NULL;
#endif
            pme->ndecompdim   = 1;
            pme->nodeid_major = pme->nodeid;
            pme->nodeid_minor = 0;

        }
        else if (nnodes_major == 1)
        {
#if GMX_MPI
            pme->mpi_comm_d[0] = MPI_COMM_NULL;
            pme->mpi_comm_d[1] = pme->mpi_comm;
#endif
            pme->ndecompdim   = 1;
            pme->nodeid_major = 0;
            pme->nodeid_minor = pme->nodeid;
        }
        else
        {
            if (pme->nnodes % nnodes_major != 0)
            {
                gmx_incons("For 2D PME decomposition, #PME ranks must be divisible by the number of ranks in the major dimension");
            }
            pme->ndecompdim = 2;

#if GMX_MPI
            MPI_Comm_split(pme->mpi_comm, pme->nodeid % nnodes_minor,
                           pme->nodeid, &pme->mpi_comm_d[0]);  /* My communicator along major dimension */
            MPI_Comm_split(pme->mpi_comm, pme->nodeid/nnodes_minor,
                           pme->nodeid, &pme->mpi_comm_d[1]);  /* My communicator along minor dimension */

            MPI_Comm_rank(pme->mpi_comm_d[0], &pme->nodeid_major);
            MPI_Comm_size(pme->mpi_comm_d[0], &pme->nnodes_major);
            MPI_Comm_rank(pme->mpi_comm_d[1], &pme->nodeid_minor);
            MPI_Comm_size(pme->mpi_comm_d[1], &pme->nnodes_minor);
#endif
        }
        pme->bPPnode = (cr->duty & DUTY_PP);
    }

    pme->bGPU = bPMEGPU && (pme->nodeid == 0);
    // only a single rank should do PME GPU currently - small steps
    // currently PME GPU mdrun with MPI crashes anyway :(

    pme->bGPUSingle = pme->bGPU && (pme->nnodes == 1);
    // a convenience variable

    pme->bGPUFFT = pme->bGPUSingle;
    // currently cuFFT is only used for a single rank
    // some Internet people have succeeded in MPI cuFFT, but I dare not venture there - Iupinov
    //yupinov this variable doesn't actually work :(

    pme->nthread = (pme->bGPU) ? 1 : nthread;

    /* Check if any of the PME MPI ranks uses threads */
    use_threads = (pme->nthread > 1 ? 1 : 0);
#if GMX_MPI
    if (pme->nnodes > 1)
    {
        MPI_Allreduce(&use_threads, &sum_use_threads, 1, MPI_INT,
                      MPI_SUM, pme->mpi_comm);
    }
    else
#endif
    {
        sum_use_threads = use_threads;
    }
    pme->bUseThreads = (sum_use_threads > 0);

    if (ir->ePBC == epbcSCREW)
    {
        gmx_fatal(FARGS, "pme does not (yet) work with pbc = screw");
    }

    pme->bFEP_q      = ((ir->efep != efepNO) && bFreeEnergy_q);
    pme->bFEP_lj     = ((ir->efep != efepNO) && bFreeEnergy_lj);
    pme->bFEP        = (pme->bFEP_q || pme->bFEP_lj);
    pme->nkx         = ir->nkx;
    pme->nky         = ir->nky;
    pme->nkz         = ir->nkz;
    pme->bP3M        = (ir->coulombtype == eelP3M_AD || getenv("GMX_PME_P3M") != NULL);
    pme->pme_order   = ir->pme_order;

    /* Always constant electrostatics coefficients */
    pme->epsilon_r   = ir->epsilon_r;

    /* Always constant LJ coefficients */
    pme->ljpme_combination_rule = ir->ljpme_combination_rule;

    /* If we violate restrictions, generate a fatal error here */
    gmx_pme_check_restrictions(pme->pme_order,
                               pme->nkx, pme->nky, pme->nkz,
                               pme->nnodes_major,
                               pme->nnodes_minor,
                               pme->bUseThreads,
                               TRUE,
                               NULL);

    if (pme->nnodes > 1)
    {
        double imbal;

#if GMX_MPI
        MPI_Type_contiguous(DIM, GMX_MPI_REAL, &(pme->rvec_mpi));
        MPI_Type_commit(&(pme->rvec_mpi));
#endif

        /* Note that the coefficient spreading and force gathering, which usually
         * takes about the same amount of time as FFT+solve_pme,
         * is always fully load balanced
         * (unless the coefficient distribution is inhomogeneous).
         */

        imbal = estimate_pme_load_imbalance(pme);
        if (imbal >= 1.2 && pme->nodeid_major == 0 && pme->nodeid_minor == 0)
        {
            fprintf(stderr,
                    "\n"
                    "NOTE: The load imbalance in PME FFT and solve is %d%%.\n"
                    "      For optimal PME load balancing\n"
                    "      PME grid_x (%d) and grid_y (%d) should be divisible by #PME_ranks_x (%d)\n"
                    "      and PME grid_y (%d) and grid_z (%d) should be divisible by #PME_ranks_y (%d)\n"
                    "\n",
                    (int)((imbal-1)*100 + 0.5),
                    pme->nkx, pme->nky, pme->nnodes_major,
                    pme->nky, pme->nkz, pme->nnodes_minor);
        }
    }

    /* For non-divisible grid we need pme_order iso pme_order-1 */
    /* In sum_qgrid_dd x overlap is copied in place: take padding into account.
     * y is always copied through a buffer: we don't need padding in z,
     * but we do need the overlap in x because of the communication order.
     */
    init_overlap_comm(&pme->overlap[0], pme->pme_order,
#if GMX_MPI
                      pme->mpi_comm_d[0],
#endif
                      pme->nnodes_major, pme->nodeid_major,
                      pme->nkx,
                      (div_round_up(pme->nky, pme->nnodes_minor)+pme->pme_order)*(pme->nkz+pme->pme_order-1));

    /* Along overlap dim 1 we can send in multiple pulses in sum_fftgrid_dd.
     * We do this with an offset buffer of equal size, so we need to allocate
     * extra for the offset. That's what the (+1)*pme->nkz is for.
     */
    init_overlap_comm(&pme->overlap[1], pme->pme_order,
#if GMX_MPI
                      pme->mpi_comm_d[1],
#endif
                      pme->nnodes_minor, pme->nodeid_minor,
                      pme->nky,
                      (div_round_up(pme->nkx, pme->nnodes_major)+pme->pme_order+1)*pme->nkz);

    /* Double-check for a limitation of the (current) sum_fftgrid_dd code.
     * Note that gmx_pme_check_restrictions checked for this already.
     */
    if (pme->bUseThreads && pme->overlap[0].noverlap_nodes > 1)
    {
        gmx_incons("More than one communication pulse required for grid overlap communication along the major dimension while using threads");
    }

    snew(pme->bsp_mod[XX], pme->nkx);
    snew(pme->bsp_mod[YY], pme->nky);
    snew(pme->bsp_mod[ZZ], pme->nkz);

    /* The required size of the interpolation grid, including overlap.
     * The allocated size (pmegrid_n?) might be slightly larger.
     */
    pme->pmegrid_nx = pme->overlap[0].s2g1[pme->nodeid_major] -
        pme->overlap[0].s2g0[pme->nodeid_major];
    pme->pmegrid_ny = pme->overlap[1].s2g1[pme->nodeid_minor] -
        pme->overlap[1].s2g0[pme->nodeid_minor];
    pme->pmegrid_nz_base = pme->nkz;
    pme->pmegrid_nz      = pme->pmegrid_nz_base + pme->pme_order - 1;
    set_grid_alignment(&pme->pmegrid_nz, pme->pme_order);
    if (pme->bGPU)
    {
        const int alignment = 32; //warp_size;
        //yupinov : if Z is not aligned by warp_size, results are wrong in PME GPU
        // where did I make this assumption?
        // so now I use same grid for all the steps on GPU (with inplace cuFFT)
        pme->pmegrid_nz = (pme->pmegrid_nz + alignment - 1) / alignment * alignment;
    }
    pme->pmegrid_start_ix = pme->overlap[0].s2g0[pme->nodeid_major];
    pme->pmegrid_start_iy = pme->overlap[1].s2g0[pme->nodeid_minor];
    pme->pmegrid_start_iz = 0;

    make_gridindex5_to_localindex(pme->nkx,
                                  pme->pmegrid_start_ix,
                                  pme->pmegrid_nx - (pme->pme_order-1),
                                  &pme->nnx, &pme->fshx);
    make_gridindex5_to_localindex(pme->nky,
                                  pme->pmegrid_start_iy,
                                  pme->pmegrid_ny - (pme->pme_order-1),
                                  &pme->nny, &pme->fshy);
    make_gridindex5_to_localindex(pme->nkz,
                                  pme->pmegrid_start_iz,
                                  pme->pmegrid_nz_base,
                                  &pme->nnz, &pme->fshz);

    pme->spline_work = make_pme_spline_work(pme->pme_order);

    ndata[0]    = pme->nkx;
    ndata[1]    = pme->nky;
    ndata[2]    = pme->nkz;
    /* It doesn't matter if we allocate too many grids here,
     * we only allocate and use the ones we need.
     */
    if (EVDW_PME(ir->vdwtype))
    {
        pme->ngrids = ((ir->ljpme_combination_rule == eljpmeLB) ? DO_Q_AND_LJ_LB : DO_Q_AND_LJ);
    }
    else
    {
        pme->ngrids = DO_Q;
    }
    snew(pme->fftgrid, pme->ngrids);
    snew(pme->cfftgrid, pme->ngrids);
    snew(pme->pfft_setup, pme->ngrids);
    if (pme->bGPU)
        snew(pme->pfft_setup_gpu, pme->ngrids); //yupinov destroy!

    for (i = 0; i < pme->ngrids; ++i)
    {
        if ((i <  DO_Q && EEL_PME(ir->coulombtype) && (i == 0 ||
                                                       bFreeEnergy_q)) ||
            (i >= DO_Q && EVDW_PME(ir->vdwtype) && (i == 2 ||
                                                    bFreeEnergy_lj ||
                                                    ir->ljpme_combination_rule == eljpmeLB)))
        {
            pmegrids_init(&pme->pmegrid[i],
                          pme->pmegrid_nx, pme->pmegrid_ny, pme->pmegrid_nz,
                          pme->pmegrid_nz_base,
                          pme->pme_order,
                          pme->bUseThreads,
                          pme->nthread,
                          pme->overlap[0].s2g1[pme->nodeid_major]-pme->overlap[0].s2g0[pme->nodeid_major+1],
                          pme->overlap[1].s2g1[pme->nodeid_minor]-pme->overlap[1].s2g0[pme->nodeid_minor+1]);
            /* This routine will allocate the grid data to fit the FFTs */
            //if (!pme->bGPU)
                gmx_parallel_3dfft_init(&pme->pfft_setup[i], ndata,
                                    &pme->fftgrid[i], &pme->cfftgrid[i],
                                    pme->mpi_comm_d,
                                     bReproducible, pme->nthread);

        }
    }

    if (!pme->bP3M)
    {
        /* Use plain SPME B-spline interpolation */
        make_bspline_moduli(pme->bsp_mod, pme->nkx, pme->nky, pme->nkz, pme->pme_order);
    }
    else
    {
        /* Use the P3M grid-optimized influence function */
        make_p3m_bspline_moduli(pme->bsp_mod, pme->nkx, pme->nky, pme->nkz, pme->pme_order);
    }

    /* Use atc[0] for spreading */
    init_atomcomm(pme, &pme->atc[0], nnodes_major > 1 ? 0 : 1, TRUE);
    if (pme->ndecompdim >= 2)
    {
        init_atomcomm(pme, &pme->atc[1], 1, FALSE);
    }

    if (pme->nnodes == 1)
    {
        pme->atc[0].n = homenr;
        pme_realloc_atomcomm_things(&pme->atc[0]);
    }

    pme_gpu_init(&pme->gpu, pme, hwinfo, gpu_opt);

    pme->lb_buf1       = NULL;
    pme->lb_buf2       = NULL;
    pme->lb_buf_nalloc = 0;

    pme_init_all_work(&pme->solve_work, pme->nthread, pme->nkx);

    *pmedata = pme;
    return 0;
}

int gmx_pme_reinit(struct gmx_pme_t **pmedata,
                   t_commrec *        cr,
                   struct gmx_pme_t * pme_src,
                   const t_inputrec * ir,
                   ivec               grid_size)
{
    t_inputrec irc;
    int        homenr;
    int        ret;

    irc     = *ir;
    irc.nkx = grid_size[XX];
    irc.nky = grid_size[YY];
    irc.nkz = grid_size[ZZ];

    if (pme_src->nnodes == 1)
    {
        homenr = pme_src->atc[0].n;
    }
    else
    {
        homenr = -1;
    }

    ret = gmx_pme_init(pmedata, cr, pme_src->nnodes_major, pme_src->nnodes_minor,
                       &irc, homenr, pme_src->bFEP_q, pme_src->bFEP_lj, FALSE, pme_src->nthread, pme_src->bGPU);

    if (ret == 0)
    {
        /* We can easily reuse the allocated pme grids in pme_src */
        reuse_pmegrids(&pme_src->pmegrid[PME_GRID_QA], &(*pmedata)->pmegrid[PME_GRID_QA]);
        /* We would like to reuse the fft grids, but that's harder */
    }

    return ret;
}

void gmx_pme_calc_energy(struct gmx_pme_t *pme, int n, rvec *x, real *q, real *V)
{
    pme_atomcomm_t *atc;
    pmegrids_t     *grid;

    if (pme->nnodes > 1)
    {
        gmx_incons("gmx_pme_calc_energy called in parallel");
    }
    if (pme->bFEP_q > 1)
    {
        gmx_incons("gmx_pme_calc_energy with free energy");
    }

    atc            = &pme->atc_energy;
    atc->nthread   = 1;
    if (atc->spline == NULL)
    {
        snew(atc->spline, atc->nthread);
    }
    atc->nslab     = 1;
    atc->bSpread   = TRUE;
    atc->pme_order = pme->pme_order;
    atc->n         = n;
    pme_realloc_atomcomm_things(atc);
    atc->x           = x;
    atc->coefficient = q;

    /* We only use the A-charges grid */
    grid = &pme->pmegrid[PME_GRID_QA];

    /* Only calculate the spline coefficients, don't actually spread */
    spread_on_grid(pme, atc, NULL, TRUE, FALSE, pme->fftgrid[PME_GRID_QA], FALSE, PME_GRID_QA, NULL); //yupinov

    *V = gather_energy_bsplines(pme, grid->grid.grid, atc);
}

/*! \brief Calculate initial Lorentz-Berthelot coefficients for LJ-PME */
static void
calc_initial_lb_coeffs(struct gmx_pme_t *pme, real *local_c6, real *local_sigma)
{
    int  i;
    for (i = 0; i < pme->atc[0].n; ++i)
    {
        real sigma4;
        sigma4                     = local_sigma[i];
        sigma4                     = sigma4*sigma4;
        sigma4                     = sigma4*sigma4;
        pme->atc[0].coefficient[i] = local_c6[i] / sigma4;
    }
}

/*! \brief Calculate next Lorentz-Berthelot coefficients for LJ-PME */
static void
calc_next_lb_coeffs(struct gmx_pme_t *pme, real *local_sigma)
{
    int  i;

    for (i = 0; i < pme->atc[0].n; ++i)
    {
        pme->atc[0].coefficient[i] *= local_sigma[i];
    }
}

int gmx_pme_do(struct gmx_pme_t *pme,
               int start,       int homenr,
               rvec x[],        rvec f[],
               real chargeA[],  real chargeB[],
               real c6A[],      real c6B[],
               real sigmaA[],   real sigmaB[],
               matrix box,      t_commrec *cr,
               int  maxshift_x, int maxshift_y,
               t_nrnb *nrnb,    gmx_wallcycle_t wcycle,
               matrix vir_q,    real ewaldcoeff_q,
               matrix vir_lj,   real ewaldcoeff_lj,
               real *energy_q,  real *energy_lj,
               real lambda_q,   real lambda_lj,
               real *dvdlambda_q, real *dvdlambda_lj,
               int flags)
{
    int                  d, i, j, npme, grid_index, max_grid_index;
    int                  n_d;
    pme_atomcomm_t      *atc        = NULL;
    pmegrids_t          *pmegrid    = NULL;
    real                *grid       = NULL;
    rvec                *f_d;
    real                *coefficient = NULL;
    real                 energy_AB[4];
    matrix               vir_AB[4];
    real                 scale, lambda;
    gmx_bool             bClearF;
    //gmx_parallel_3dfft_t pfft_setup;
    real              *  fftgrid;
    t_complex          * cfftgrid;
    int                  thread;
    gmx_bool             bFirst, bDoSplines;
    int                  fep_state;
    int                  fep_states_lj           = pme->bFEP_lj ? 2 : 1;
    const gmx_bool       bCalcEnerVir            = flags & GMX_PME_CALC_ENER_VIR;
    const gmx_bool       bBackFFT                = flags & (GMX_PME_CALC_F | GMX_PME_CALC_POT);
    const gmx_bool       bCalcF                  = flags & GMX_PME_CALC_F;

    assert(pme->nnodes > 0);
    assert(pme->nnodes == 1 || pme->ndecompdim > 0);

    if (pme->nnodes > 1)
    {
        atc      = &pme->atc[0];
        atc->npd = homenr;
        if (atc->npd > atc->pd_nalloc)
        {
            atc->pd_nalloc = over_alloc_dd(atc->npd);
            srenew(atc->pd, atc->pd_nalloc);
        }
        for (d = pme->ndecompdim-1; d >= 0; d--)
        {
            atc           = &pme->atc[d];
            atc->maxshift = (atc->dimind == 0 ? maxshift_x : maxshift_y);
        }
    }
    else
    {
        atc = &pme->atc[0];
        /* This could be necessary for TPI */
        pme->atc[0].n = homenr;
        if (DOMAINDECOMP(cr))
        {
            pme_realloc_atomcomm_things(atc);
        }
        atc->x = x;
        atc->f = f;
    }

    gmx::invertBoxMatrix(box, pme->recipbox);

    bFirst = TRUE;
    pme_gpu_step_init(pme);

    /* For simplicity, we construct the splines for all particles if
     * more than one PME calculations is needed. Some optimization
     * could be done by keeping track of which atoms have splines
     * constructed, and construct new splines on each pass for atoms
     * that don't yet have them.
     */

    bDoSplines = pme->bFEP || ((flags & GMX_PME_DO_COULOMB) && (flags & GMX_PME_DO_LJ));

    /* We need a maximum of four separate PME calculations:
     * grid_index=0: Coulomb PME with charges from state A
     * grid_index=1: Coulomb PME with charges from state B
     * grid_index=2: LJ PME with C6 from state A
     * grid_index=3: LJ PME with C6 from state B
     * For Lorentz-Berthelot combination rules, a separate loop is used to
     * calculate all the terms
     */

    /* If we are doing LJ-PME with LB, we only do Q here */
    max_grid_index = (pme->ljpme_combination_rule == eljpmeLB) ? DO_Q : DO_Q_AND_LJ;

    if (pme->bGPU)
    {
        GMX_RELEASE_ASSERT(!(flags & GMX_PME_DO_LJ) && !pme->bFEP, "PME GPU has only been tried for a single grid. Shouldn't be difficult to extend though.\n");
        GMX_RELEASE_ASSERT(sizeof(real) == sizeof(float), "PME GPU was not been designed with double precision in mind.\nIt might be possible too implement, but would require meticulous code proofreading.\n");
    }
    for (grid_index = 0; grid_index < max_grid_index; ++grid_index)
    {
        /* Check if we should do calculations at this grid_index
         * If grid_index is odd we should be doing FEP
         * If grid_index < 2 we should be doing electrostatic PME
         * If grid_index >= 2 we should be doing LJ-PME
         */
        if ((grid_index <  DO_Q && (!(flags & GMX_PME_DO_COULOMB) ||
                                    (grid_index == 1 && !pme->bFEP_q))) ||
            (grid_index >= DO_Q && (!(flags & GMX_PME_DO_LJ) ||
                                    (grid_index == 3 && !pme->bFEP_lj))))
        {
            continue;
        }
        /* Unpack structure */
        pmegrid    = &pme->pmegrid[grid_index];
        fftgrid    = pme->fftgrid[grid_index];
        cfftgrid   = pme->cfftgrid[grid_index];
        //pfft_setup = pme->pfft_setup[grid_index];
        switch (grid_index)
        {
            case 0: coefficient = chargeA + start; break;
            case 1: coefficient = chargeB + start; break;
            case 2: coefficient = c6A + start; break;
            case 3: coefficient = c6B + start; break;
        }

        grid = pmegrid->grid.grid;

        if (debug)
        {
            fprintf(debug, "PME: number of ranks = %d, rank = %d\n",
                    cr->nnodes, cr->nodeid);
            fprintf(debug, "Grid = %p\n", (void*)grid);
            if (grid == NULL)
            {
                gmx_fatal(FARGS, "No grid!");
            }
        }
        where();

        if (pme->nnodes == 1)
        {
            atc->coefficient = coefficient;
        }
        else
        {
            wallcycle_start(wcycle, ewcPME_REDISTXF);
            do_redist_pos_coeffs(pme, cr, start, homenr, bFirst, x, coefficient);
            where();

            wallcycle_stop(wcycle, ewcPME_REDISTXF);
        }

        if (pme->bGPU)
            pme_gpu_copy_charges(pme); //yupinov

        if (debug)
        {
            fprintf(debug, "Rank= %6d, pme local particles=%6d\n",
                    cr->nodeid, atc->n);
        }
        gmx_bool keepGPUDataBetweenSpreadAndR2C = FALSE;
        gmx_bool keepGPUDataBetweenC2RAndGather = FALSE; //yupinov inconvenient
        if (pme->bGPU)
        {
            //yupinov - these are not checked anywhere yet
            //check for spread and solve flags here as well!
            // bGPUSingle
            //yupinov set to true to segfault
            keepGPUDataBetweenSpreadAndR2C = pme->bGPUSingle && pme->bGPUFFT; //yupinov -> no wrap kernels! different grids!
            gmx_bool keepGPUDataBetweenR2CAndSolve = pme->bGPUSingle && pme->bGPUFFT && (grid_index < DO_Q); // no LJ support
            gmx_bool keepGPUDataBetweenSolveAndC2R = pme->bGPUSingle && keepGPUDataBetweenR2CAndSolve && bBackFFT;
            keepGPUDataBetweenC2RAndGather = pme->bGPUSingle && pme->bGPUFFT; // bCalcF!
            pme_gpu_update_flags(pme->gpu,
                                 keepGPUDataBetweenSpreadAndR2C,
                                 keepGPUDataBetweenR2CAndSolve,
                                 keepGPUDataBetweenSolveAndC2R,
                                 keepGPUDataBetweenC2RAndGather
                                 );
        }

        if (flags & GMX_PME_SPREAD)
        {
            wallcycle_start(wcycle, ewcPME_SPREADGATHER);

            /* Spread the coefficients on a grid */
            spread_on_grid(pme, &pme->atc[0], pmegrid, bFirst, TRUE, fftgrid, bDoSplines, grid_index, wcycle);

            if (bFirst)
            {
                inc_nrnb(nrnb, eNR_WEIGHTS, DIM*atc->n);
            }
            inc_nrnb(nrnb, eNR_SPREADBSP,
                     pme->pme_order*pme->pme_order*pme->pme_order*atc->n);

            if (!pme->bUseThreads)
            {
                if (!pme->bGPUSingle) // only wrap CPU PME grid if we haven't done it on GPU in a single GPU mode
                    wrap_periodic_pmegrid(pme, grid); //yupinov - unwrap as well

                /* sum contributions to local grid from other nodes */
#if GMX_MPI
                if (pme->nnodes > 1)
                {
                    gmx_sum_qgrid_dd(pme, grid, GMX_SUM_GRID_FORWARD);
                    where();
                }
#endif
                if (!keepGPUDataBetweenSpreadAndR2C) // only copy CPU PME grid to CPU FFT grid if we don't keep the data on GPU
                    copy_pmegrid_to_fftgrid(pme, grid, fftgrid, grid_index);
            }

            wallcycle_stop(wcycle, ewcPME_SPREADGATHER);

            /* TODO If the OpenMP and single-threaded implementations
               converge, then spread_on_grid() and
               copy_pmegrid_to_fftgrid() will perhaps live in the same
               source file and the following debugging function can live
               there too. */

             /*
               dump_local_fftgrid(pme,fftgrid);
               exit(0);
             */
        }

        /* Here we start a large thread parallel region */
#pragma omp parallel num_threads(pme->nthread) private(thread)
        {
            try
            {
                thread = gmx_omp_get_thread_num();
                if (flags & GMX_PME_SOLVE)
                {
                    int loop_count;

                    /*
                    #pragma omp barrier
                    if (thread == 0)
                        dump_local_fftgrid(pme, fftgrid, grid_index);
                    #pragma omp barrier
                    */

                    /* do 3d-fft */
                    gmx_parallel_3dfft_execute_wrapper(pme, grid_index, GMX_FFT_REAL_TO_COMPLEX,
                                               thread, wcycle);
                    where();

                    /*
                    #pragma omp barrier
                    if (thread == 0)
                        dump_local_fftgrid(pme, (const real *)cfftgrid, grid_index);
                    #pragma omp barrier
                    */


                    /* solve in k-space for our local cells */
                    if (thread == 0)
                    {
                        wallcycle_start(wcycle, (grid_index < DO_Q ? ewcPME_SOLVE : ewcLJPME));
                        wallcycle_sub_start(wcycle, ewcsPME_SOLVE);
                    }
                    if (grid_index < DO_Q)
                    {
                        loop_count =
                            solve_pme_yzx_wrapper(pme, cfftgrid, ewaldcoeff_q,
                                          box[XX][XX]*box[YY][YY]*box[ZZ][ZZ],
                                          bCalcEnerVir,
                                          pme->nthread, thread);
                         //if (pme->bGPU && bCalcEnerVir)
                         //   pme_gpu_get_energy_virial(pme);
                    }
                    else
                    {
                        loop_count =
                            solve_pme_lj_yzx_wrapper(pme, &cfftgrid, FALSE, ewaldcoeff_lj,
                                             box[XX][XX]*box[YY][YY]*box[ZZ][ZZ],
                                             bCalcEnerVir,
                                             pme->nthread, thread);  //yupinov implement LJ
                    }
                    /*
                    #pragma omp barrier
                    if (thread == 0)
                        dump_local_fftgrid(pme, (const real *)cfftgrid, grid_index);
                    #pragma omp barrier
                    */
                    if (thread == 0)
                    {
                        wallcycle_stop(wcycle, (grid_index < DO_Q ? ewcPME_SOLVE : ewcLJPME));
                        wallcycle_sub_stop(wcycle, ewcsPME_SOLVE);
                        where();
                        inc_nrnb(nrnb, eNR_SOLVEPME, loop_count);
                    }
                }

                if (bBackFFT)
                {
                    /* do 3d-invfft */
                    if (thread == 0)
                    {
                        where();
                    }
                    gmx_parallel_3dfft_execute_wrapper(pme, grid_index, GMX_FFT_COMPLEX_TO_REAL,
                                              thread, wcycle);
                    /*
                    #pragma omp barrier
                    if (thread == 0)
                        dump_local_fftgrid(pme,(const real *)fftgrid, grid_index);
                    #pragma omp barrier
                    */

                    if (thread == 0)
                    {
                        where();

                        if (pme->nodeid == 0)
                        {
                            real ntot = pme->nkx*pme->nky*pme->nkz;
                            npme  = static_cast<int>(ntot*std::log(ntot)/std::log(2.0));
                            inc_nrnb(nrnb, eNR_FFT, 2*npme);
                        }

                        /* Note: this wallcycle region is closed below
                           outside an OpenMP region, so take care if
                           refactoring code here. */
                        wallcycle_start(wcycle, ewcPME_SPREADGATHER);
                    }
                    if (!keepGPUDataBetweenC2RAndGather)
                        copy_fftgrid_to_pmegrid(pme, fftgrid, grid, grid_index, pme->nthread, thread);
                }
            } GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
        }
        /* End of thread parallel section.
         * With MPI we have to synchronize here before gmx_sum_qgrid_dd.
         */

        if (bBackFFT)
        {
            /* distribute local grid to all nodes */
#if GMX_MPI
            if (pme->nnodes > 1)
            {
                gmx_sum_qgrid_dd(pme, grid, GMX_SUM_GRID_BACKWARD);
            }
#endif
            where();
            if (!pme->bGPUSingle) //yupinov check GPUFFT instead?
                unwrap_periodic_pmegrid(pme, grid);
        }

        if (bCalcF)
        {
            /* interpolate forces for our local atoms */

            where();

            /* If we are running without parallelization,
             * atc->f is the actual force array, not a buffer,
             * therefore we should not clear it.
             */
            lambda  = grid_index < DO_Q ? lambda_q : lambda_lj;
            bClearF = (bFirst && PAR(cr));
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
            for (thread = 0; thread < pme->nthread; thread++)
            {
                try
                {
                    gather_f_bsplines_wrapper(pme, grid, bClearF, atc,
                                      &atc->spline[thread],
                                      pme->bFEP ? (grid_index % 2 == 0 ? 1.0-lambda : lambda) : 1.0, wcycle, thread);
                }
                GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
            }
            /*
            #pragma omp barrier
            dump_local_fftgrid(pme, (const real *)grid, grid_index);
            #pragma omp barrier
            */
            where();

            //if (pme->bGPU)
            //    gpu_forces_copyback(pme, atc->n, atc->f); //yupinov fix compacted particle count

            inc_nrnb(nrnb, eNR_GATHERFBSP,
                     pme->pme_order*pme->pme_order*pme->pme_order*pme->atc[0].n);
            /* Note: this wallcycle region is opened above inside an OpenMP
               region, so take care if refactoring code here. */
            wallcycle_stop(wcycle, ewcPME_SPREADGATHER);
        }

        if (bCalcEnerVir)
        {
            /* This should only be called on the master thread
             * and after the threads have synchronized.
             */
            if (grid_index < 2)
            {
                get_pme_ener_vir_q(pme->solve_work, pme->nthread, &energy_AB[grid_index], vir_AB[grid_index]);
            }
            else
            {
                get_pme_ener_vir_lj(pme->solve_work, pme->nthread, &energy_AB[grid_index], vir_AB[grid_index]);
            }
        }
        bFirst = FALSE;
    } /* of grid_index-loop */

    /* For Lorentz-Berthelot combination rules in LJ-PME, we need to calculate
     * seven terms. */

    if ((flags & GMX_PME_DO_LJ) && pme->ljpme_combination_rule == eljpmeLB)
    {
        /* Loop over A- and B-state if we are doing FEP */
        for (fep_state = 0; fep_state < fep_states_lj; ++fep_state)
        {
            real *local_c6 = NULL, *local_sigma = NULL, *RedistC6 = NULL, *RedistSigma = NULL;
            if (pme->nnodes == 1)
            {
                if (pme->lb_buf1 == NULL)
                {
                    pme->lb_buf_nalloc = pme->atc[0].n;
                    snew(pme->lb_buf1, pme->lb_buf_nalloc);
                }
                pme->atc[0].coefficient = pme->lb_buf1;
                switch (fep_state)
                {
                    case 0:
                        local_c6      = c6A;
                        local_sigma   = sigmaA;
                        break;
                    case 1:
                        local_c6      = c6B;
                        local_sigma   = sigmaB;
                        break;
                    default:
                        gmx_incons("Trying to access wrong FEP-state in LJ-PME routine");
                }
            }
            else
            {
                atc = &pme->atc[0];
                switch (fep_state)
                {
                    case 0:
                        RedistC6      = c6A;
                        RedistSigma   = sigmaA;
                        break;
                    case 1:
                        RedistC6      = c6B;
                        RedistSigma   = sigmaB;
                        break;
                    default:
                        gmx_incons("Trying to access wrong FEP-state in LJ-PME routine");
                }
                wallcycle_start(wcycle, ewcPME_REDISTXF);

                do_redist_pos_coeffs(pme, cr, start, homenr, bFirst, x, RedistC6);
                if (pme->lb_buf_nalloc < atc->n)
                {
                    pme->lb_buf_nalloc = atc->nalloc;
                    srenew(pme->lb_buf1, pme->lb_buf_nalloc);
                    srenew(pme->lb_buf2, pme->lb_buf_nalloc);
                }
                local_c6 = pme->lb_buf1;
                for (i = 0; i < atc->n; ++i)
                {
                    local_c6[i] = atc->coefficient[i];
                }
                where();

                do_redist_pos_coeffs(pme, cr, start, homenr, FALSE, x, RedistSigma);
                local_sigma = pme->lb_buf2;
                for (i = 0; i < atc->n; ++i)
                {
                    local_sigma[i] = atc->coefficient[i];
                }
                where();

                wallcycle_stop(wcycle, ewcPME_REDISTXF);
            }
            calc_initial_lb_coeffs(pme, local_c6, local_sigma);

            /*Seven terms in LJ-PME with LB, grid_index < 2 reserved for electrostatics*/
            for (grid_index = 2; grid_index < 9; ++grid_index)
            {
                /* Unpack structure */
                pmegrid    = &pme->pmegrid[grid_index];
                fftgrid    = pme->fftgrid[grid_index];
                //pfft_setup = pme->pfft_setup[grid_index];
                calc_next_lb_coeffs(pme, local_sigma);
                grid = pmegrid->grid.grid;
                where();

                if (flags & GMX_PME_SPREAD)
                {
                    wallcycle_start(wcycle, ewcPME_SPREADGATHER);
                    /* Spread the c6 on a grid */
                    spread_on_grid(pme, &pme->atc[0], pmegrid, bFirst, TRUE, fftgrid, bDoSplines, grid_index, wcycle);

                    if (bFirst)
                    {
                        inc_nrnb(nrnb, eNR_WEIGHTS, DIM*atc->n);
                    }

                    inc_nrnb(nrnb, eNR_SPREADBSP,
                             pme->pme_order*pme->pme_order*pme->pme_order*atc->n);
                    if (pme->nthread == 1)
                    {
                        wrap_periodic_pmegrid(pme, grid);
                        /* sum contributions to local grid from other nodes */
#if GMX_MPI
                        if (pme->nnodes > 1)
                        {
                            gmx_sum_qgrid_dd(pme, grid, GMX_SUM_GRID_FORWARD);
                            where();
                        }
#endif
                        copy_pmegrid_to_fftgrid(pme, grid, fftgrid, grid_index);
                    }
                    wallcycle_stop(wcycle, ewcPME_SPREADGATHER);
                }
                /*Here we start a large thread parallel region*/
#pragma omp parallel num_threads(pme->nthread) private(thread)
                {
                    try
                    {
                        thread = gmx_omp_get_thread_num();
                        if (flags & GMX_PME_SOLVE)
                        {
                            /* do 3d-fft */
                            gmx_parallel_3dfft_execute_wrapper(pme, grid_index, GMX_FFT_REAL_TO_COMPLEX,
                                                       thread, wcycle);
                            where();
                        }
                    }
                    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
                }
                bFirst = FALSE;
            }
            if (flags & GMX_PME_SOLVE)
            {
                /* solve in k-space for our local cells */
#pragma omp parallel num_threads(pme->nthread) private(thread)
                {
                    try
                    {
                        int loop_count;
                        thread = gmx_omp_get_thread_num();
                        if (thread == 0)
                        {
                            wallcycle_start(wcycle, ewcLJPME);
                        }

                        loop_count =
                            solve_pme_lj_yzx(pme, &pme->cfftgrid[2], TRUE, ewaldcoeff_lj,
                                             box[XX][XX]*box[YY][YY]*box[ZZ][ZZ],
                                             bCalcEnerVir,
                                             pme->nthread, thread);
                        if (thread == 0)
                        {
                            wallcycle_stop(wcycle, ewcLJPME);
                            where();
                            inc_nrnb(nrnb, eNR_SOLVEPME, loop_count);
                        }
                    }
                    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
                }
            }

            if (bCalcEnerVir)
            {
                /* This should only be called on the master thread and
                 * after the threads have synchronized.
                 */
                get_pme_ener_vir_lj(pme->solve_work, pme->nthread, &energy_AB[2+fep_state], vir_AB[2+fep_state]);
            }

            if (bBackFFT)
            {
                bFirst = !(flags & GMX_PME_DO_COULOMB);
                calc_initial_lb_coeffs(pme, local_c6, local_sigma);
                for (grid_index = 8; grid_index >= 2; --grid_index)
                {
                    /* Unpack structure */
                    pmegrid    = &pme->pmegrid[grid_index];
                    fftgrid    = pme->fftgrid[grid_index];
                    //pfft_setup = pme->pfft_setup[grid_index];
                    grid       = pmegrid->grid.grid;
                    calc_next_lb_coeffs(pme, local_sigma);
                    where();
#pragma omp parallel num_threads(pme->nthread) private(thread)
                    {
                        try
                        {
                            thread = gmx_omp_get_thread_num();
                            /* do 3d-invfft */
                            if (thread == 0)
                            {
                                where();
                            }

                            gmx_parallel_3dfft_execute_wrapper(pme, grid_index, GMX_FFT_COMPLEX_TO_REAL,
                                                       thread, wcycle);
                            if (thread == 0)
                            {
                                where();

                                if (pme->nodeid == 0)
                                {
                                    real ntot = pme->nkx*pme->nky*pme->nkz;
                                    npme  = static_cast<int>(ntot*std::log(ntot)/std::log(2.0));
                                    inc_nrnb(nrnb, eNR_FFT, 2*npme);
                                }
                                wallcycle_start(wcycle, ewcPME_SPREADGATHER);
                            }

                            copy_fftgrid_to_pmegrid(pme, fftgrid, grid, grid_index, pme->nthread, thread);
                        }
                        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
                    } /*#pragma omp parallel*/

                    /* distribute local grid to all nodes */
#if GMX_MPI
                    if (pme->nnodes > 1)
                    {
                        gmx_sum_qgrid_dd(pme, grid, GMX_SUM_GRID_BACKWARD);
                    }
#endif
                    where();

                    unwrap_periodic_pmegrid(pme, grid);

                    if (bCalcF)
                    {
                        /* interpolate forces for our local atoms */
                        where();
                        bClearF = (bFirst && PAR(cr));
                        scale   = pme->bFEP ? (fep_state < 1 ? 1.0-lambda_lj : lambda_lj) : 1.0;
                        scale  *= lb_scale_factor[grid_index-2];

#pragma omp parallel for num_threads(pme->nthread) schedule(static)
                        for (thread = 0; thread < pme->nthread; thread++)
                        {
                            try
                            {
                                gather_f_bsplines_wrapper(pme, grid, bClearF, &pme->atc[0],
                                                  &pme->atc[0].spline[thread],
                                                  scale, wcycle, thread);
                            }
                            GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
                        }

                        where();

                        inc_nrnb(nrnb, eNR_GATHERFBSP,
                                 pme->pme_order*pme->pme_order*pme->pme_order*pme->atc[0].n);
                    }
                    //if (pme->bGPU)
                    //    gpu_forces_copyback(pme, atc->n, atc->f); //yupinov fix compacted particle count

                    wallcycle_stop(wcycle, ewcPME_SPREADGATHER);

                    bFirst = FALSE;
                } /* for (grid_index = 8; grid_index >= 2; --grid_index) */
            }     /* if (bCalcF) */
        }         /* for (fep_state = 0; fep_state < fep_states_lj; ++fep_state) */
    }             /* if ((flags & GMX_PME_DO_LJ) && pme->ljpme_combination_rule == eljpmeLB) */

    pme_gpu_step_end(pme, bCalcF, bCalcEnerVir);

    if (pme->bGPU) // whole body copied from up there in the loop...
    {
        const int grid_index = 0;
        if (bCalcEnerVir)
        {
            /* This should only be called on the master thread
             * and after the threads have synchronized.
             */
            if (grid_index < 2)
            {
                get_pme_ener_vir_q(pme->solve_work, pme->nthread, &energy_AB[grid_index], vir_AB[grid_index]);
            }
            else
            {
                get_pme_ener_vir_lj(pme->solve_work, pme->nthread, &energy_AB[grid_index], vir_AB[grid_index]);
            }
        }
    }

    // debug
    /*
    rvec test = {0, 0, 0};
    for (int i = 0; i < pme->atc[0].n; i++)
        if (memcmp(test, pme->atc[0].f[i], sizeof(rvec)))
        {
            printf("FOUND something %d %g %g %g\n", i, pme->atc[0].f[i][XX], pme->atc[0].f[i][YY], pme->atc[0].f[i][ZZ]);
            break;
        }

    printf("%g\n", energy_AB[0]);
    */
    // debug end

    if (bCalcF && pme->nnodes > 1)
    {
        wallcycle_start(wcycle, ewcPME_REDISTXF);
        for (d = 0; d < pme->ndecompdim; d++)
        {
            atc = &pme->atc[d];
            if (d == pme->ndecompdim - 1)
            {
                n_d = homenr;
                f_d = f + start;
            }
            else
            {
                n_d = pme->atc[d+1].n;
                f_d = pme->atc[d+1].f;
            }
            if (DOMAINDECOMP(cr))
            {
                dd_pmeredist_f(pme, atc, n_d, f_d,
                               d == pme->ndecompdim-1 && pme->bPPnode);
            }
        }

        wallcycle_stop(wcycle, ewcPME_REDISTXF);
    }
    where();

    if (bCalcEnerVir)
    {
        if (flags & GMX_PME_DO_COULOMB)
        {
            if (!pme->bFEP_q)
            {
                *energy_q = energy_AB[0];
                m_add(vir_q, vir_AB[0], vir_q);
            }
            else
            {
                *energy_q       = (1.0-lambda_q)*energy_AB[0] + lambda_q*energy_AB[1];
                *dvdlambda_q   += energy_AB[1] - energy_AB[0];
                for (i = 0; i < DIM; i++)
                {
                    for (j = 0; j < DIM; j++)
                    {
                        vir_q[i][j] += (1.0-lambda_q)*vir_AB[0][i][j] +
                            lambda_q*vir_AB[1][i][j];
                    }
                }
            }
            if (debug)
            {
                fprintf(debug, "Electrostatic PME mesh energy: %g\n", *energy_q);
            }
        }
        else
        {
            *energy_q = 0;
        }

        if (flags & GMX_PME_DO_LJ)
        {
            if (!pme->bFEP_lj)
            {
                *energy_lj = energy_AB[2];
                m_add(vir_lj, vir_AB[2], vir_lj);
            }
            else
            {
                *energy_lj     = (1.0-lambda_lj)*energy_AB[2] + lambda_lj*energy_AB[3];
                *dvdlambda_lj += energy_AB[3] - energy_AB[2];
                for (i = 0; i < DIM; i++)
                {
                    for (j = 0; j < DIM; j++)
                    {
                        vir_lj[i][j] += (1.0-lambda_lj)*vir_AB[2][i][j] + lambda_lj*vir_AB[3][i][j];
                    }
                }
            }
            if (debug)
            {
                fprintf(debug, "Lennard-Jones PME mesh energy: %g\n", *energy_lj);
            }
        }
        else
        {
            *energy_lj = 0;
        }
    }
    return 0;
}
