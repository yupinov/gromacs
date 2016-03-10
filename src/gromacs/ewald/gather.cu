#include "pme.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include <cuda.h>

#include "th-a.cuh"
#include "check.h"

#include "pme-internal.h"
#include "pme-cuda.h"

typedef real *splinevec[DIM];

#ifdef DEBUG_PME_TIMINGS_GPU
extern gpu_events gpu_events_gather;
#endif

#define DO_FSPLINE(order)                      \
    for (int ithx = 0; (ithx < order); ithx++)              \
    {                                              \
        const int index_x = (i0[globalIndex] + ithx) * pny * pnz;               \
        const real tx = thx[thetaOffset + ithx];                       \
        const real dx = dthx[thetaOffset + ithx];                      \
                                               \
        for (int ithy = 0; (ithy < order); ithy++)          \
        {                                          \
            const int index_xy = index_x+(j0[globalIndex]+ithy)*pnz;      \
            const real ty = thy[thetaOffset + ithy];                  \
            const real dy = dthy[thetaOffset + ithy];                 \
            real fxy1 = 0.0f; \
            real fz1 = 0.0f;		   \
                                               \
            /*for (int ithz = 0; (ithz < order); ithz++)    */  \
            /*   gridValue[particlesPerBlock * ithz + localIndex] = grid[index_xy+(k0[globalIndex]+ithz)];*/\
            for (int ithz = 0; (ithz < order); ithz++)      \
            {                                      \
                /*printf(" INDEX %d %d %d\n", (i0[i] + ithx), (j0[i]+ithy), (k0[i]+ithz));*/\
                /*gridValue[localIndex] = grid[index_xy+(k0[globalIndex]+ithz)]; */ \
                /*fxy1 += thz[thetaOffset + ithz] * gridValue[particlesPerBlock * ithz + localIndex];  */          \
                /*fz1  += dthz[thetaOffset + ithz] * gridValue[particlesPerBlock * ithz + localIndex];    */       \
                const real gridValue = grid[index_xy+(k0[globalIndex]+ithz)];  \
                fxy1 += thz[thetaOffset + ithz] * gridValue; \
                fz1  += dthz[thetaOffset + ithz] * gridValue; \
            }                                      \
            fx[localIndex] += dx * ty * fxy1;                      \
            fy[localIndex] += tx * dy * fxy1;                      \
            fz[localIndex] += tx * ty * fz1;                       \
        }                                          \
    }

template <const int particlesPerBlock, const int order>
//__launch_bounds__(4 * warp_size, 16)
//yupinov - with this, on my GTX 660 Ti, occupancy is 0.84, but it's slower by what, 20%?
//same for minblocks = 14
//without it, it's faster, but occupancy is 0.52 out of 62.5
static __global__ void gather_f_bsplines_kernel
(const real * __restrict__ grid, const int n,
 const int nx, const int ny, const int nz, const int pnx, const int pny, const int pnz,
 const real rxx, const real ryx, const real ryy, const real rzx, const real rzy, const real rzz,
 const real * __restrict__ thx, const real * __restrict__ thy, const real * __restrict__ thz,
 const real * __restrict__ dthx, const real * __restrict__ dthy, const real * __restrict__ dthz,
 real * __restrict__ atc_f, const real * __restrict__ coefficient_v,
 const int * __restrict__ i0, const int * __restrict__ j0, const int * __restrict__ k0)
{
    /* sum forces for local particles */
    const int localIndex = threadIdx.x;
    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ real fx[particlesPerBlock];
    __shared__ real fy[particlesPerBlock];
    __shared__ real fz[particlesPerBlock];
    __shared__ real coefficient[particlesPerBlock];

    //__shared__ real gridValue[order * particlesPerBlock];

    if (globalIndex < n)
    {
        coefficient[localIndex] = coefficient_v[globalIndex];
        fx[localIndex] = 0.0f;
        fy[localIndex] = 0.0f;
        fz[localIndex] = 0.0f;
        const int thetaOffset = globalIndex * order;
        const int idim = globalIndex * DIM;

        switch (order)
        {
            case 4:
                DO_FSPLINE(4);
                break;
            case 5:
                DO_FSPLINE(5);
                break;
            default:
                DO_FSPLINE(order);
                break;
        }

        atc_f[idim + XX] += -coefficient[localIndex] * ( fx[localIndex] * nx * rxx );
        atc_f[idim + YY] += -coefficient[localIndex] * ( fx[localIndex] * nx * ryx + fy[localIndex] * ny * ryy );
        atc_f[idim + ZZ] += -coefficient[localIndex] * ( fx[localIndex] * nx * rzx + fy[localIndex] * ny * rzy + fz[localIndex] * nz * rzz );

        /* Since the energy and not forces are interpolated
         * the net force might not be exactly zero.
         * This can be solved by also interpolating F, but
         * that comes at a cost.
         * A better hack is to remove the net force every
         * step, but that must be done at a higher level
         * since this routine doesn't see all atoms if running
         * in parallel. Don't know how important it is?  EL 990726
         */
    }
}

void gather_f_bsplines_gpu_2_pre
(gmx_bool bClearF,
 int *spline_ind, int spline_n,
 real *atc_coefficient, rvec *atc_f,
 real scale, int thread
 )
{
    // compact atc_f before cpu calcucation

    int size_forces = DIM * spline_n * sizeof(real);
    real *atc_f_compacted = th_a(TH_ID_F, thread, size_forces, TH_LOC_HOST); //yupinov fixed allocation size - not actually compacted, same for i_compacted
    int size_indices = spline_n * sizeof(int);
    int *atc_i_compacted = th_i(TH_ID_I, thread, size_indices, TH_LOC_HOST);

    int oo = 0;
    for (int ii = 0; ii < spline_n; ii++)
    {
        int i           = spline_ind[ii];
        real coefficient_i = scale*atc_coefficient[i];
        if (bClearF)
        {
            atc_f[i][XX] = 0;
            atc_f[i][YY] = 0;
            atc_f[i][ZZ] = 0;
        }

        if (coefficient_i != 0.0f)
        {
            atc_f_compacted[oo * DIM + XX] = atc_f[i][XX];
            atc_f_compacted[oo * DIM + YY] = atc_f[i][YY];
            atc_f_compacted[oo * DIM + ZZ] = atc_f[i][ZZ];
            atc_i_compacted[oo] = i;  // indices of uncompacted particles stored in a compacted array
            oo++;
        }
    }
    //oo is a real size of compacted stuff now
}

void gather_f_bsplines_gpu_2
(real *grid, gmx_bool bClearF,
 const int order,
 int nx, int ny, int nz, int pnx, int pny, int pnz,
 real rxx, real ryx, real ryy, real rzx, real rzy, real rzz,
 int *spline_ind, int spline_n,
 real *atc_coefficient, rvec *atc_f, ivec *atc_idx,
 splinevec *spline_theta, splinevec *spline_dtheta,
 real scale,
 gmx_pme_t *pme,
 int thread
 )
{
    cudaStream_t s = pme->gpu->pmeStream;
    int ndatatot = pnx*pny*pnz;

    if (!spline_n)
        return;

    int size_grid = ndatatot * sizeof(real);
    real *grid_d = th_a_cpy(TH_ID_GRID, thread, grid, size_grid, TH_LOC_CUDA, s);

    //copy order?
    //compacting, and size....
    int n = spline_n;
    int size_indices = n * sizeof(int);
    int size_coefficients = n * sizeof(real);
    int size_forces = DIM * n * sizeof(real);
    int size_splines = order * n * sizeof(int);


    real *atc_f_compacted = th_a(TH_ID_F, thread, -1, TH_LOC_HOST); //but that's wrong! realloc

    int *atc_i_compacted = th_i(TH_ID_I, thread, -1, TH_LOC_HOST);  //way to get sizes from th-a?
    real *coefficients_compacted = th_a(TH_ID_COEFFICIENT, thread, size_coefficients, TH_LOC_HOST);
    //yupinov reuse H_ID_COEFFICIENT and other stuff from before solve?

    int *i0_compacted = th_i(TH_ID_I0, thread, size_indices, TH_LOC_HOST); //yupinov these are IDXPTR, actually. maybe split it?
    int *j0_compacted = th_i(TH_ID_J0, thread, size_indices, TH_LOC_HOST);
    int *k0_compacted = th_i(TH_ID_K0, thread, size_indices, TH_LOC_HOST);

    real *theta_x_compacted = th_a(TH_ID_THX, thread, size_splines, TH_LOC_HOST);
    real *theta_y_compacted = th_a(TH_ID_THY, thread, size_splines, TH_LOC_HOST);
    real *theta_z_compacted = th_a(TH_ID_THZ, thread, size_splines, TH_LOC_HOST);
    real *dtheta_x_compacted = th_a(TH_ID_DTHX, thread, size_splines, TH_LOC_HOST);
    real *dtheta_y_compacted = th_a(TH_ID_DTHY, thread, size_splines, TH_LOC_HOST);
    real *dtheta_z_compacted = th_a(TH_ID_DTHZ, thread, size_splines, TH_LOC_HOST);

    int oo = 0;
    for (int ii = 0; ii < spline_n; ii++)
    {
        int i           = spline_ind[ii];
        real coefficient_i = scale*atc_coefficient[i];
        if (bClearF)
        {
            atc_f[i][XX] = 0; //yupinov memset?
            atc_f[i][YY] = 0;
            atc_f[i][ZZ] = 0;
        }

        if (coefficient_i != 0)
        {
            coefficients_compacted[oo] = coefficient_i;
            int *idxptr = atc_idx[i];
            //Mattias: atc_f_h force-copying is in gather_f_bsplines_gpu_2_pre()
            //yupinov: the fuck is it doing there?
            atc_i_compacted[oo] = i;
            i0_compacted[oo] = idxptr[XX];
            j0_compacted[oo] = idxptr[YY];
            k0_compacted[oo] = idxptr[ZZ];
            int iiorder = ii*order;
            int ooorder = oo*order;
            for (int o = 0; o < order; ++o)
            {
                theta_x_compacted[ooorder + o] = (*spline_theta)[XX][iiorder + o];
                theta_y_compacted[ooorder + o] = (*spline_theta)[YY][iiorder + o];
                theta_z_compacted[ooorder + o] = (*spline_theta)[ZZ][iiorder + o];
                dtheta_x_compacted[ooorder + o] = (*spline_dtheta)[XX][iiorder + o];
                dtheta_y_compacted[ooorder + o] = (*spline_dtheta)[YY][iiorder + o];
                dtheta_z_compacted[ooorder + o] = (*spline_dtheta)[ZZ][iiorder + o];
            }
            ++oo;
        }
    }

    n = oo;
    if (!n)
        return;

    //copypasted
    size_indices = n * sizeof(int);
    size_coefficients = n * sizeof(real);
    size_forces = DIM * n * sizeof(real);
    size_splines = order * n * sizeof(int);

    real *atc_f_d = th_a_cpy(TH_ID_F, thread, atc_f_compacted, size_forces, TH_LOC_CUDA, s);
    real *coefficients_d = th_a_cpy(TH_ID_COEFFICIENT, thread, coefficients_compacted, size_coefficients, TH_LOC_CUDA, s);

    int *i0_d = th_i_cpy(TH_ID_I0, thread, i0_compacted, size_indices, TH_LOC_CUDA, s);
    int *j0_d = th_i_cpy(TH_ID_J0, thread, j0_compacted, size_indices, TH_LOC_CUDA, s);
    int *k0_d = th_i_cpy(TH_ID_K0, thread, k0_compacted, size_indices, TH_LOC_CUDA, s);

    real *theta_x_d = th_a_cpy(TH_ID_THX, thread, theta_x_compacted, size_splines, TH_LOC_CUDA, s);
    real *theta_y_d = th_a_cpy(TH_ID_THY, thread, theta_y_compacted, size_splines, TH_LOC_CUDA, s);
    real *theta_z_d = th_a_cpy(TH_ID_THZ, thread, theta_z_compacted, size_splines, TH_LOC_CUDA, s);
    real *dtheta_x_d = th_a_cpy(TH_ID_DTHX, thread, dtheta_x_compacted, size_splines, TH_LOC_CUDA, s);
    real *dtheta_y_d = th_a_cpy(TH_ID_DTHY, thread, dtheta_y_compacted, size_splines, TH_LOC_CUDA, s);
    real *dtheta_z_d = th_a_cpy(TH_ID_DTHZ, thread, dtheta_z_compacted, size_splines, TH_LOC_CUDA, s);

    const int blockSize = 4 * warp_size;
    int n_blocks = (n + blockSize - 1) / blockSize;
#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_start(gpu_events_gather, s);
#endif
    if (order == 4) //yupinov
        gather_f_bsplines_kernel<blockSize, 4> <<<n_blocks, blockSize, 0, s>>>
          (grid_d,
           n,
           nx, ny, nz, pnx, pny, pnz,
           rxx, ryx, ryy, rzx, rzy, rzz,
           theta_x_d, theta_y_d, theta_z_d,
           dtheta_x_d, dtheta_y_d, dtheta_z_d,
           atc_f_d, coefficients_d,
           i0_d, j0_d, k0_d);
    CU_LAUNCH_ERR("gather_f_bsplines_kernel");
#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_stop(gpu_events_gather, s, ewcsPME_GATHER, 0);
#endif

    th_cpy(atc_f_compacted, atc_f_d, size_forces, TH_LOC_HOST, s);

    for (int ii = 0; ii < n; ii++)  // iterating over compacted particles
    {
        int i = atc_i_compacted[ii]; //index of uncompacted particle
        atc_f[i][XX] = atc_f_compacted[ii * DIM + XX];
        atc_f[i][YY] = atc_f_compacted[ii * DIM + YY];
        atc_f[i][ZZ] = atc_f_compacted[ii * DIM + ZZ];
    }
}
