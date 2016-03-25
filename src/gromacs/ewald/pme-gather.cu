#include "pme.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include <cuda.h>

#include "pme-timings.cuh"

#include "pme-internal.h"
#include "pme-cuda.cuh"

#include <assert.h>

#define SHARED_MEMORY_REDUCTION 1

//yupinov - texture memory?
template <
        const int order,
        const int particlesPerBlock
        >
__launch_bounds__(4 * warp_size, 16)
static __global__ void pme_gather_kernel
(const real * __restrict__ grid, const int n,
 const int nx, const int ny, const int nz, const int pnx, const int pny, const int pnz,
 const real rxx, const real ryx, const real ryy, const real rzx, const real rzy, const real rzz,
 const real * __restrict__ thx, const real * __restrict__ thy, const real * __restrict__ thz,
 const real * __restrict__ dthx, const real * __restrict__ dthy, const real * __restrict__ dthz,
 real * __restrict__ atc_f, const real * __restrict__ coefficient_v,
 //const int * __restrict__ i0, const int * __restrict__ j0, const int * __restrict__ k0,
 const int * __restrict__ idx
 )
{
    /* sum forces for local particles */

    // these are particle indices - in shared and global memory
    const int localIndex = threadIdx.z;
    const int globalIndex = blockIdx.x * blockDim.z + threadIdx.z;

    const int particleDataSize = order * order;
    const int blockSize = particlesPerBlock * particleDataSize; //1 line per thread
    // with odd orders something might break here?


    // spline Y/Z coordinates
    const int ithy = threadIdx.y;
    const int ithz = threadIdx.x;
    // these are spline contribution indices in shared memory
    const int splineIndex = threadIdx.y * blockDim.x + threadIdx.x;   // relative to the current particle
    const int lineIndex = (threadIdx.z * (blockDim.x * blockDim.y)) + splineIndex; // and to all the block's particles

    const int idxSize = DIM * particlesPerBlock;
    __shared__ int sharedIdx[idxSize];

    int blockId = blockIdx.x
                 + blockIdx.y * gridDim.x
                 + gridDim.x * gridDim.y * blockIdx.z;
    /*
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                  + (threadIdx.z * (blockDim.x * blockDim.y))
                  + (threadIdx.y * blockDim.x)
                  + threadIdx.x;
                  */
    int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;
    if (threadLocalId < idxSize)
    {
        sharedIdx[threadLocalId] = idx[blockIdx.x * idxSize + threadLocalId];
    }//locality?
    __syncthreads();


#if SHARED_MEMORY_REDUCTION
    __shared__ real fx[blockSize];
    __shared__ real fy[blockSize];
    __shared__ real fz[blockSize];
    fx[lineIndex] = 0.0f;
    fy[lineIndex] = 0.0f;
    fz[lineIndex] = 0.0f;
#else
    real fx = 0.0f;
    real fy = 0.0f;
    real fz = 0.0f;
#endif

    __shared__ real coefficient[particlesPerBlock];

    if (globalIndex < n)
    {
        const int thetaOffset = globalIndex * order;

        for (int ithx = 0; (ithx < order); ithx++)
        {
            //const int index_x = (i0[globalIndex] + ithx) * pny * pnz;
            //const int index_x = (idx[globalIndex * DIM + XX] + ithx) * pny * pnz;
            const int index_x = (sharedIdx[localIndex * DIM + XX] + ithx) * pny * pnz;
            //if (blockId == 1)
            //    printf("%d %d\n", idx[globalIndex * DIM + XX], sharedIdx[localIndex * DIM + XX]);

            const real tx = thx[thetaOffset + ithx];
            const real dx = dthx[thetaOffset + ithx];

            //for (int ithy = 0; (ithy < order); ithy++)
            {
                //const int index_xy = index_x + (j0[globalIndex] + ithy) * pnz;
                //const int index_xy = index_x + (idx[globalIndex * DIM + YY] + ithy) * pnz;
                const int index_xy = index_x + (sharedIdx[localIndex * DIM + YY] + ithy) * pnz;
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
                    //const real gridValue = grid[index_xy + (k0[globalIndex] + ithz)];
                    //const real gridValue = grid[index_xy + (idx[globalIndex * DIM + ZZ] + ithz)];
                    const real gridValue = grid[index_xy + (sharedIdx[localIndex * DIM + ZZ] + ithz)];
                    fxy1 += thz[thetaOffset + ithz] * gridValue;
                    fz1  += dthz[thetaOffset + ithz] * gridValue;
                }
                //yupinov do a normal reduction here and below
#if SHARED_MEMORY_REDUCTION
                fx[lineIndex] += dx * ty * fxy1;
                fy[lineIndex] += tx * dy * fxy1;
                fz[lineIndex] += tx * ty * fz1;
#else
                fx += dx * ty * fxy1;
                fy += tx * dy * fxy1;
                fz += tx * ty * fz1;
#endif
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
    // skip shared memory,
    //  do a shuffle loop stopping before last step for order 4



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
        coefficient[localIndex] = coefficient_v[globalIndex];
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

template <
    const int order,
    const int stage
    >
__global__ void pme_unwrap_kernel
    (const int nx, const int ny, const int nz,
     const int pnx,const int pny, const int pnz,
     real * __restrict__ grid)
{
    //UNWRAP

    const int iz = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int ix = blockIdx.z * blockDim.z + threadIdx.z;

    const int overlap = order - 1;

    int    ny_x;//, ix;

    //if (pme->nnodes_major == 1)
    if (stage & 4)
    {
        //ny_x = (pme->nnodes_minor == 1 ? ny : pme->pmegrid_ny);
        ny_x = ny;

        if (iz < nz)
        //for (ix = 0; ix < overlap; ix++)
        {
            //for (iy = 0; iy < ny_x; iy++)
            {
                //for (iz = 0; iz < nz; iz++)
                {
                    const int address = (ix * pny + iy) * pnz + iz;
                    const int offset_x = nx * pny * pnz;
                    grid[address + offset_x] = grid[address];
                }
            }
        }
    }

    //if (pme->nnodes_minor == 1)
    if (stage & 2)
    {
        if (iz < nz)
        //for (ix = 0; ix < pnx; ix++)
        {
            //int iy, iz;

            //for (iy = 0; iy < overlap; iy++)
            {
                //for (iz = 0; iz < nz; iz++)
                {
                    const int address = (ix * pny + iy) * pnz + iz;
                    const int offset_y = ny * pnz;
                    grid[address + offset_y] = grid[address];
                }
            }
        }
    }

    /* Copy periodic overlap in z */
    if (stage & 1)
    {
        //for (ix = 0; ix < pnx; ix++)
        if (iy < pny)
        {
            //int iy, iz;

            //for (iy = 0; iy < pny; iy++)
            {
                //for (iz = 0; iz < overlap; iz++)
                {
                    const int address = (ix * pny + iy) * pnz + iz;
                    const int offset_z = nz;
                    grid[address + offset_z] = grid[address];
                }
            }
        }
    }
}

void gather_f_bsplines_gpu
(real *grid, gmx_bool bClearF,
 const int order,
 int nx, int ny, int nz, int pnx, int pny, int pnz,
 real rxx, real ryx, real ryy, real rzx, real rzy, real rzz,
 int *spline_ind, int n,
 real *atc_coefficient, rvec *atc_f, ivec *atc_idx,
 splinevec *spline_theta, splinevec *spline_dtheta,
 real scale,
 gmx_pme_t *pme,
 int thread
 )
{
    cudaStream_t s = pme->gpu->pmeStream;
    if (!n)
        return;

    const int ndatatot = pnx * pny * pnz;
    const int gridSize = ndatatot * sizeof(real);
    real *grid_d = PMEFetchRealArray(PME_ID_REAL_GRID, thread, gridSize, ML_DEVICE);
    if (!pme->gpu->keepGPUDataBetweenC2RAndGather)
        PMECopy(grid_d, grid, gridSize, ML_DEVICE, s);

    if (pme->bGPUSingle)
    {
        if (order == 4)
        {
            const int blockSize = 4 * warp_size; //yupinov thsi is everywhere! and arichitecture-specific
            const int overlap = order - 1; // all copied from pme-spread.cu
            int overlapLinesPerBlock = blockSize / overlap; //so there is unused padding in each block;

            dim3 blocks[] =
            {
                dim3(1, (pny + overlapLinesPerBlock - 1) / overlapLinesPerBlock, pnx),
                dim3((nz + overlapLinesPerBlock - 1) / overlapLinesPerBlock, 1, pnx),
                dim3((nz + overlapLinesPerBlock - 1) / overlapLinesPerBlock, ny, 1),
            };
            // low occupancy :(
            dim3 threads[] =
            {
                dim3(overlap, overlapLinesPerBlock, 1),
                dim3(overlapLinesPerBlock, overlap, 1),
                dim3(overlapLinesPerBlock, 1, overlap),
            };

            events_record_start(gpu_events_unwrap, s);

            pme_unwrap_kernel<4, 4> <<<blocks[2], threads[2], 0, s>>>(nx, ny, nz, pnx, pny, pnz, grid_d);
            pme_unwrap_kernel<4, 2> <<<blocks[1], threads[1], 0, s>>>(nx, ny, nz, pnx, pny, pnz, grid_d);
            pme_unwrap_kernel<4, 1> <<<blocks[0], threads[0], 0, s>>>(nx, ny, nz, pnx, pny, pnz, grid_d);

            CU_LAUNCH_ERR("pme_unwrap_kernel");

            events_record_stop(gpu_events_unwrap, s, ewcsPME_UNWRAP, 0);
        }
        else
            gmx_fatal(FARGS, "gather: orders other than 4 untested!");
    }

    int size_forces = DIM * n * sizeof(real); //yupinov!
    int size_indices = n * sizeof(int);
    int size_splines = order * n * sizeof(int);
    int size_coefficients = n * sizeof(real);

    real *atc_f_h = NULL;
    int *i0_h = NULL, *j0_h = NULL, *k0_h = NULL;
    real *coefficients_h = NULL;

    real *theta_x_h = NULL, *theta_y_h = NULL, *theta_z_h = NULL;
    real *dtheta_x_h = NULL, *dtheta_y_h = NULL, *dtheta_z_h = NULL;

    //indices - allocated here because maybe different sturcture?
    i0_h = PMEFetchIntegerArray(PME_ID_I0, thread, size_indices, ML_HOST);
    j0_h = PMEFetchIntegerArray(PME_ID_J0, thread, size_indices, ML_HOST);
    k0_h = PMEFetchIntegerArray(PME_ID_K0, thread, size_indices, ML_HOST);
    //yupinov broken!

    int *atc_i_compacted_h = NULL;


    // compact data
    if (PME_SKIP_ZEROES)
    {
        atc_i_compacted_h = PMEFetchIntegerArray(PME_ID_NONZERO_INDICES, thread, size_indices, ML_HOST);

        // fixed host allocation sizes - will only be smaller on GPU

        // forces
        atc_f_h = PMEFetchRealArray(PME_ID_F, thread, size_forces, ML_HOST);

        // thetas
        theta_x_h = PMEFetchRealArray(PME_ID_THX, thread, size_splines, ML_HOST);
        theta_y_h = PMEFetchRealArray(PME_ID_THY, thread, size_splines, ML_HOST);
        theta_z_h = PMEFetchRealArray(PME_ID_THZ, thread, size_splines, ML_HOST);
        dtheta_x_h = PMEFetchRealArray(PME_ID_DTHX, thread, size_splines, ML_HOST);
        dtheta_y_h = PMEFetchRealArray(PME_ID_DTHY, thread, size_splines, ML_HOST);
        dtheta_z_h = PMEFetchRealArray(PME_ID_DTHZ, thread, size_splines, ML_HOST);

        // coefficients
        coefficients_h = PMEFetchRealArray(PME_ID_COEFFICIENT, thread, size_coefficients, ML_HOST);

        int iCompacted = 0;
        for (int ii = 0; ii < n; ii++)
        {
            int iOriginal = spline_ind[ii]; //yupinov is there a point to this spline_ind? shoould be just 1 : 1

            assert(spline_ind[ii] == ii);

            // coefficients
            real coefficient_i = scale * atc_coefficient[iOriginal]; //yupinov mutiply coefficients on device!

            if (coefficient_i != 0.0f)
            {
                coefficients_h[iCompacted] = coefficient_i;

                //indices
                int *idxptr = atc_idx[iOriginal];
                i0_h[iCompacted] = idxptr[XX];
                j0_h[iCompacted] = idxptr[YY];
                k0_h[iCompacted] = idxptr[ZZ];

                //thetas
                int iiorder = ii * order;
                int ooorder = iCompacted * order;
                for (int o = 0; o < order; ++o)
                {
                    theta_x_h[ooorder + o] = (*spline_theta)[XX][iiorder + o];
                    theta_y_h[ooorder + o] = (*spline_theta)[YY][iiorder + o];
                    theta_z_h[ooorder + o] = (*spline_theta)[ZZ][iiorder + o];
                    dtheta_x_h[ooorder + o] = (*spline_dtheta)[XX][iiorder + o];
                    dtheta_y_h[ooorder + o] = (*spline_dtheta)[YY][iiorder + o];
                    dtheta_z_h[ooorder + o] = (*spline_dtheta)[ZZ][iiorder + o];
                }

                //forces
                if (!bClearF)
                {
                    atc_f_h[iCompacted * DIM + XX] = atc_f[iOriginal][XX];
                    atc_f_h[iCompacted * DIM + YY] = atc_f[iOriginal][YY];
                    atc_f_h[iCompacted * DIM + ZZ] = atc_f[iOriginal][ZZ];
                }

                // indices of uncompacted particles stored in a compacted array
                atc_i_compacted_h[iCompacted] = iOriginal;

                iCompacted++;
            }
        }
        // adjust sizes for device allocation
        n = iCompacted;
        size_coefficients = n * sizeof(real);
        size_splines = order * n * sizeof(int);
        size_indices = n * sizeof(int);
        size_forces = DIM * n * sizeof(real);
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            // indices
            i0_h[i] = atc_idx[i][XX]; //yupinov reorganize
            j0_h[i] = atc_idx[i][YY];
            k0_h[i] = atc_idx[i][ZZ];

            // coefficients
            atc_coefficient[i] *= scale;
        }

        // forces
        atc_f_h = (real *)atc_f;
        // coefficients
        coefficients_h = atc_coefficient;
        // thetas
        theta_x_h = (*spline_theta)[XX];
        theta_y_h = (*spline_theta)[YY];
        theta_z_h = (*spline_theta)[ZZ];
        dtheta_x_h = (*spline_dtheta)[XX];
        dtheta_y_h = (*spline_dtheta)[YY];
        dtheta_z_h = (*spline_dtheta)[ZZ];
    }

    // thetas
    /*
    real *theta_x_d = PMEFetchAndCopyRealArray(PME_ID_THX, thread, theta_x_h, size_splines, ML_DEVICE, s);
    real *theta_y_d = PMEFetchAndCopyRealArray(PME_ID_THY, thread, theta_y_h, size_splines, ML_DEVICE, s);
    real *theta_z_d = PMEFetchAndCopyRealArray(PME_ID_THZ, thread, theta_z_h, size_splines, ML_DEVICE, s);
    real *dtheta_x_d = PMEFetchAndCopyRealArray(PME_ID_DTHX, thread, dtheta_x_h, size_splines, ML_DEVICE, s);
    real *dtheta_y_d = PMEFetchAndCopyRealArray(PME_ID_DTHY, thread, dtheta_y_h, size_splines, ML_DEVICE, s);
    real *dtheta_z_d = PMEFetchAndCopyRealArray(PME_ID_DTHZ, thread, dtheta_z_h, size_splines, ML_DEVICE, s);
    */
    real *theta_d = PMEFetchRealArray(PME_ID_THETA, thread, DIM * size_splines, ML_DEVICE);
    real *theta_x_d = theta_d + 0 * order * n;
    real *theta_y_d = theta_d + 1 * order * n;
    real *theta_z_d = theta_d + 2 * order * n;

    real *dtheta_d = PMEFetchRealArray(PME_ID_DTHETA, thread, DIM * size_splines, ML_DEVICE);
    real *dtheta_x_d = dtheta_d + 0 * order * n;
    real *dtheta_y_d = dtheta_d + 1 * order * n;
    real *dtheta_z_d = dtheta_d + 2 * order * n;

    // coefficients
    real *coefficients_d = PMEFetchRealArray(PME_ID_COEFFICIENT, thread, size_coefficients, ML_DEVICE);

    int *idx_d = PMEFetchIntegerArray(PME_ID_IDXPTR, thread, DIM * size_indices, ML_DEVICE);

    if (!pme->gpu->keepGPUDataBetweenC2RAndGather) // compare with spread and compacting
    {
        PMECopy(theta_x_d, theta_x_h, size_splines, ML_DEVICE, s);
        PMECopy(theta_y_d, theta_y_h, size_splines, ML_DEVICE, s);
        PMECopy(theta_z_d, theta_z_h, size_splines, ML_DEVICE, s);

        PMECopy(dtheta_x_d, dtheta_x_h, size_splines, ML_DEVICE, s);
        PMECopy(dtheta_y_d, dtheta_y_h, size_splines, ML_DEVICE, s);
        PMECopy(dtheta_z_d, dtheta_z_h, size_splines, ML_DEVICE, s);

        PMECopy(coefficients_d, coefficients_h, size_coefficients, ML_DEVICE, s);

        PMECopy(idx_d, atc_idx, DIM * size_indices, ML_DEVICE, s);
    }

    //forces
    real *atc_f_d = PMEFetchRealArray(PME_ID_F, thread, size_forces, ML_DEVICE);
    if (bClearF)
    {
        cudaError_t stat = cudaMemsetAsync(atc_f_d, 0, size_forces, s);
        CU_RET_ERR(stat, "cudaMemsetAsync gather forces error");
    }
    else
        PMECopy(atc_f_d, atc_f_h, size_forces, ML_DEVICE, s);

    //indices

    //yupinov
    /*
    int *i0_d = PMEFetchAndCopyIntegerArray(PME_ID_I0, thread, i0_h, size_indices, ML_DEVICE, s);
    int *j0_d = PMEFetchAndCopyIntegerArray(PME_ID_J0, thread, j0_h, size_indices, ML_DEVICE, s);
    int *k0_d = PMEFetchAndCopyIntegerArray(PME_ID_K0, thread, k0_h, size_indices, ML_DEVICE, s);
    */


    const int blockSize = 4 * warp_size;
    const int particlesPerBlock = blockSize / order / order;
    dim3 nBlocks((n + blockSize - 1) / blockSize * order * order, 1, 1); //yupinov what does this mean?
    dim3 dimBlock(order, order, particlesPerBlock);

    events_record_start(gpu_events_gather, s);

    if (order == 4) //yupinov
        pme_gather_kernel<4, blockSize / 4 / 4> <<<nBlocks, dimBlock, 0, s>>>
          (grid_d,
           n,
           nx, ny, nz, pnx, pny, pnz,
           rxx, ryx, ryy, rzx, rzy, rzz,
           theta_x_d, theta_y_d, theta_z_d,
           dtheta_x_d, dtheta_y_d, dtheta_z_d,
           atc_f_d, coefficients_d,
           //i0_d, j0_d, k0_d,
           idx_d);
    else
        gmx_fatal(FARGS, "gather: orders other than 4 untested!");
    CU_LAUNCH_ERR("pme_gather_kernel");

    events_record_stop(gpu_events_gather, s, ewcsPME_GATHER, 0);

    PMECopy(atc_f_h, atc_f_d, size_forces, ML_HOST, s);

    if (PME_SKIP_ZEROES)
        for (int ii = 0; ii < n; ii++)  // iterating over compacted particles
        {
            int i = atc_i_compacted_h[ii]; //index of uncompacted particle
            atc_f[i][XX] = atc_f_h[ii * DIM + XX];
            atc_f[i][YY] = atc_f_h[ii * DIM + YY];
            atc_f[i][ZZ] = atc_f_h[ii * DIM + ZZ];
        }
}
