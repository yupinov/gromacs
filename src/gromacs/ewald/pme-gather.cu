#include "pme.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include <cuda.h>

#include "check.h"

#include "pme-internal.h"
#include "pme-cuda.h"

#ifdef DEBUG_PME_TIMINGS_GPU
extern gpu_events gpu_events_gather;
#endif              \

//yupinov - texture memory?
template <
        const int order,
        const int particlesPerBlock
        >
__launch_bounds__(4 * warp_size, 16)
//yupinov - with this, on my GTX 660 Ti, occupancy is 0.84, but it's slower by what, 20%?
//same for minblocks = 14
//without it, it's faster, but occupancy is 0.52 out of 62.5
static __global__ void pme_gather_kernel
(const real * __restrict__ grid, const int n,
 const int nx, const int ny, const int nz, const int pnx, const int pny, const int pnz,
 const real rxx, const real ryx, const real ryy, const real rzx, const real rzy, const real rzz,
 const real * __restrict__ thx, const real * __restrict__ thy, const real * __restrict__ thz,
 const real * __restrict__ dthx, const real * __restrict__ dthy, const real * __restrict__ dthz,
 real * __restrict__ atc_f, const real * __restrict__ coefficient_v,
 const int * __restrict__ i0, const int * __restrict__ j0, const int * __restrict__ k0)
{
    /* sum forces for local particles */

    // these are paricle indices
    const int localIndex = threadIdx.x;
    const int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int particleDataSize = order * order;
    const int blockSize = particlesPerBlock * particleDataSize; //1 line per thread
    //yupinov -> this is actually not a full block size! with odd orders somethimg will break here!
    __shared__ real fx[blockSize];
    __shared__ real fy[blockSize];
    __shared__ real fz[blockSize];

    //__shared__ real gridValue[order * particlesPerBlock];

    // spline Y/Z coordinates
    const int ithy = threadIdx.y;
    const int ithz = threadIdx.z;
    const int splineIndex = ithy * order + ithz;
    const int lineIndex = localIndex * particleDataSize + splineIndex;

    fx[lineIndex] = 0.0f;
    fy[lineIndex] = 0.0f;
    fz[lineIndex] = 0.0f;

    __shared__ real coefficient[particlesPerBlock];

    if (globalIndex < n)
    {
        if (splineIndex == 0) //yupinov stupid
            coefficient[localIndex] = coefficient_v[globalIndex];

        const int thetaOffset = globalIndex * order;

        for (int ithx = 0; (ithx < order); ithx++)
        {
            const int index_x = (i0[globalIndex] + ithx) * pny * pnz;
            const real tx = thx[thetaOffset + ithx];
            const real dx = dthx[thetaOffset + ithx];

            //for (int ithy = 0; (ithy < order); ithy++)
            {
                const int index_xy = index_x + (j0[globalIndex] + ithy) * pnz;
                const real ty = thy[thetaOffset + ithy];
                const real dy = dthy[thetaOffset + ithy];
                real fxy1 = 0.0f;
                real fz1 = 0.0f;

                /*for (int ithz = 0; (ithz < order); ithz++)    */
                /*   gridValue[particlesPerBlock * ithz + localIndex] = grid[index_xy + (k0[globalIndex] + ithz)];*/
                //for (int ithz = 0; (ithz < order); ithz++)
                {
                    /*printf(" INDEX %d %d %d\n", (i0[i] + ithx), (j0[i]+ithy), (k0[i]+ithz));*/
                    /*gridValue[localIndex] = grid[index_xy+(k0[globalIndex]+ithz)]; */
                    /*fxy1 += thz[thetaOffset + ithz] * gridValue[particlesPerBlock * ithz + localIndex];  */
                    /*fz1  += dthz[thetaOffset + ithz] * gridValue[particlesPerBlock * ithz + localIndex];    */
                    const real gridValue = grid[index_xy + (k0[globalIndex] + ithz)];
                    fxy1 += thz[thetaOffset + ithz] * gridValue;
                    fz1  += dthz[thetaOffset + ithz] * gridValue;
                }
                //yupinov do a normal reduction here and below
                fx[lineIndex] += dx * ty * fxy1;
                fy[lineIndex] += tx * dy * fxy1;
                fz[lineIndex] += tx * ty * fz1;
                /*
                atomicAdd(fx + localIndex, dx * ty * fxy1);
                atomicAdd(fy + localIndex, tx * dy * fxy1);
                atomicAdd(fz + localIndex, tx * ty * fz1);
                */
                /*
                fx[localIndex] += dx * ty * fxy1;
                fy[localIndex] += tx * dy * fxy1;
                fz[localIndex] += tx * ty * fz1;
                */
            }
        }
    }
    __syncthreads(); // breaking globalIndex condition?

    // now particlesPerBlock have to sum order^2 contributions each

    // do a simple reduction in shared mem
    for (unsigned int s = 1; s < particleDataSize; s *= 2)//<<= 1)
    {
        if ((splineIndex % (2 * s) == 0) && (splineIndex + s < particleDataSize))
        {
            // order = 5 => splineIndex 24 (the last one) will get neighbour element without the second conditional
            // unroll for different orders?
            fx[lineIndex] += fx[lineIndex + s];
            fy[lineIndex] += fy[lineIndex + s];
            fz[lineIndex] += fz[lineIndex + s];
        }
        __syncthreads();
    }


    // below is the failed modified reduction #6
    // from http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
    // (they have even better #7!)
    /*
    if (particleDataSize >= 512)
    {
        if (splineIndex < 256)
        {
            fx[lineIndex] += fx[lineIndex + 256];
            fy[lineIndex] += fy[lineIndex + 256];
            fz[lineIndex] += fz[lineIndex + 256];
        }
        __syncthreads();
    }
    if (particleDataSize >= 256)
    {
        if (splineIndex < 128)
        {
            fx[lineIndex] += fx[lineIndex + 128];
            fy[lineIndex] += fy[lineIndex + 128];
            fz[lineIndex] += fz[lineIndex + 128];
        }
        __syncthreads();
    }
    if (particleDataSize >= 128)
    {
        if (splineIndex < 64)
        {
            fx[lineIndex] += fx[lineIndex + 64];
            fy[lineIndex] += fy[lineIndex + 64];
            fz[lineIndex] += fz[lineIndex + 64];
        }
        __syncthreads();
    }
    //if (splineIndex < 32) //yupinov this is inside-warp-magic to not sync threads anymore - brings me mistakes?
    {
        if ((particleDataSize >= 64) && (splineIndex < 32))
        {
            fx[lineIndex] += fx[lineIndex + 32];
            fy[lineIndex] += fy[lineIndex + 32];
            fz[lineIndex] += fz[lineIndex + 32];
        }
        __syncthreads();
        if ((particleDataSize >= 32) && (splineIndex < 16))
        {
            fx[lineIndex] += fx[lineIndex + 16];
            fy[lineIndex] += fy[lineIndex + 16];
            fz[lineIndex] += fz[lineIndex + 16];
        }
        __syncthreads();
        if ((particleDataSize >= 16) && (splineIndex < 8))
        {
            fx[lineIndex] += fx[lineIndex +  8];
            fy[lineIndex] += fy[lineIndex +  8];
            fz[lineIndex] += fz[lineIndex +  8];
        }
        __syncthreads();
        if ((particleDataSize >=  8) && (splineIndex < 4))
        {
            fx[lineIndex] += fx[lineIndex +  4];
            fy[lineIndex] += fy[lineIndex +  4];
            fz[lineIndex] += fz[lineIndex +  4];
        }
        __syncthreads();
        if ((particleDataSize >=  4) && (splineIndex < 2))
        {
            fx[lineIndex] += fx[lineIndex +  2];
            fy[lineIndex] += fy[lineIndex +  2];
            fz[lineIndex] += fz[lineIndex +  2];
        }
        __syncthreads();
        if ((particleDataSize >=  2) && (splineIndex == 0))
        {
            fx[lineIndex] += fx[lineIndex +  1];
            fy[lineIndex] += fy[lineIndex +  1];
            fz[lineIndex] += fz[lineIndex +  1];
        }
        __syncthreads();
    }
    */

    if (splineIndex == 0) //yupinov stupid
    {
        const int idim = globalIndex * DIM;
        const int sumIndex = localIndex * particleDataSize;
        fx[sumIndex] *= (real) nx;
        fy[sumIndex] *= (real) ny;
        fz[sumIndex] *= (real) nz;
        atc_f[idim + XX] += -coefficient[localIndex] * ( fx[sumIndex] * rxx );
        atc_f[idim + YY] += -coefficient[localIndex] * ( fx[sumIndex] * ryx + fy[sumIndex] * ryy );
        atc_f[idim + ZZ] += -coefficient[localIndex] * ( fx[sumIndex] * rzx + fy[sumIndex] * rzy + fz[sumIndex] * rzz );
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
    real *atc_f_compacted = PMEFetchRealArray(PME_ID_F, thread, size_forces, ML_HOST); //yupinov fixed allocation size - not actually compacted, same for i_compacted
    int size_indices = spline_n * sizeof(int);
    int *atc_i_compacted = PMEFetchIntegerArray(PME_ID_I, thread, size_indices, ML_HOST);

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
    real *grid_d = PMEFetchAndCopyRealArray(PME_ID_REAL_GRID_WITH_OVERLAP, thread, grid, size_grid, ML_DEVICE, s);

    //copy order?
    //compacting, and size....
    int n = spline_n;
    int size_indices = n * sizeof(int);
    int size_coefficients = n * sizeof(real);
    int size_forces = DIM * n * sizeof(real);
    int size_splines = order * n * sizeof(int);


    real *atc_f_compacted = PMEFetchRealArray(PME_ID_F, thread, -1, ML_HOST); //but that's wrong! realloc

    int *atc_i_compacted = PMEFetchIntegerArray(PME_ID_I, thread, -1, ML_HOST);  //way to get sizes from th-a?
    real *coefficients_compacted = PMEFetchRealArray(PME_ID_COEFFICIENT, thread, size_coefficients, ML_HOST);
    //yupinov reuse H_ID_COEFFICIENT and other stuff from before solve?

    int *i0_compacted = PMEFetchIntegerArray(PME_ID_I0, thread, size_indices, ML_HOST); //yupinov these are IDXPTR, actually. maybe split it?
    int *j0_compacted = PMEFetchIntegerArray(PME_ID_J0, thread, size_indices, ML_HOST);
    int *k0_compacted = PMEFetchIntegerArray(PME_ID_K0, thread, size_indices, ML_HOST);

    real *theta_x_compacted = PMEFetchRealArray(PME_ID_THX, thread, size_splines, ML_HOST);
    real *theta_y_compacted = PMEFetchRealArray(PME_ID_THY, thread, size_splines, ML_HOST);
    real *theta_z_compacted = PMEFetchRealArray(PME_ID_THZ, thread, size_splines, ML_HOST);
    real *dtheta_x_compacted = PMEFetchRealArray(PME_ID_DTHX, thread, size_splines, ML_HOST);
    real *dtheta_y_compacted = PMEFetchRealArray(PME_ID_DTHY, thread, size_splines, ML_HOST);
    real *dtheta_z_compacted = PMEFetchRealArray(PME_ID_DTHZ, thread, size_splines, ML_HOST);

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

    real *atc_f_d = PMEFetchAndCopyRealArray(PME_ID_F, thread, atc_f_compacted, size_forces, ML_DEVICE, s);
    real *coefficients_d = PMEFetchAndCopyRealArray(PME_ID_COEFFICIENT, thread, coefficients_compacted, size_coefficients, ML_DEVICE, s);

    int *i0_d = PMEFetchAndCopyIntegerArray(PME_ID_I0, thread, i0_compacted, size_indices, ML_DEVICE, s);
    int *j0_d = PMEFetchAndCopyIntegerArray(PME_ID_J0, thread, j0_compacted, size_indices, ML_DEVICE, s);
    int *k0_d = PMEFetchAndCopyIntegerArray(PME_ID_K0, thread, k0_compacted, size_indices, ML_DEVICE, s);

    real *theta_x_d = PMEFetchAndCopyRealArray(PME_ID_THX, thread, theta_x_compacted, size_splines, ML_DEVICE, s);
    real *theta_y_d = PMEFetchAndCopyRealArray(PME_ID_THY, thread, theta_y_compacted, size_splines, ML_DEVICE, s);
    real *theta_z_d = PMEFetchAndCopyRealArray(PME_ID_THZ, thread, theta_z_compacted, size_splines, ML_DEVICE, s);
    real *dtheta_x_d = PMEFetchAndCopyRealArray(PME_ID_DTHX, thread, dtheta_x_compacted, size_splines, ML_DEVICE, s);
    real *dtheta_y_d = PMEFetchAndCopyRealArray(PME_ID_DTHY, thread, dtheta_y_compacted, size_splines, ML_DEVICE, s);
    real *dtheta_z_d = PMEFetchAndCopyRealArray(PME_ID_DTHZ, thread, dtheta_z_compacted, size_splines, ML_DEVICE, s);

    const int blockSize = 4 * warp_size;
    const int particlesPerBlock = blockSize / order / order;
    dim3 nBlocks((n + blockSize - 1) / blockSize * order * order, 1, 1); //yupinov what does this mean?
    dim3 dimBlock(particlesPerBlock, order, order);
#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_start(gpu_events_gather, s);
#endif
    if (order == 4) //yupinov
        pme_gather_kernel<4, blockSize / 4 / 4> <<<nBlocks, dimBlock, 0, s>>>
          (grid_d,
           n,
           nx, ny, nz, pnx, pny, pnz,
           rxx, ryx, ryy, rzx, rzy, rzz,
           theta_x_d, theta_y_d, theta_z_d,
           dtheta_x_d, dtheta_y_d, dtheta_z_d,
           atc_f_d, coefficients_d,
           i0_d, j0_d, k0_d);
    else
        gmx_fatal(FARGS, "gather: orders other than 4 untested!");
    CU_LAUNCH_ERR("gather_f_bsplines_kernel");
#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_stop(gpu_events_gather, s, ewcsPME_GATHER, 0);
#endif

    PMECopy(atc_f_compacted, atc_f_d, size_forces, ML_HOST, s);

    for (int ii = 0; ii < n; ii++)  // iterating over compacted particles
    {
        int i = atc_i_compacted[ii]; //index of uncompacted particle
        atc_f[i][XX] = atc_f_compacted[ii * DIM + XX];
        atc_f[i][YY] = atc_f_compacted[ii * DIM + YY];
        atc_f[i][ZZ] = atc_f_compacted[ii * DIM + ZZ];
    }
}
