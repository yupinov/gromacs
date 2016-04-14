#include "pme.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/math/gmxcomplex.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"

#include <cuda.h>

#include <assert.h>

#include "pme-timings.cuh"


#include "pme-cuda.cuh"
#include "pme-gpu.h"
#include "pme-internal.h"
#include "pme-solve.h"

#define SQRT_M_PI real(2.0f / M_2_SQRTPI)

/* Pascal triangle coefficients used in solve_pme_lj_yzx, only need to do 4 calculations due to symmetry */
static const __constant__ real lb_scale_factor_symm_gpu[] = { 2.0/64, 12.0/64, 30.0/64, 20.0/64 };
// copied from pme-internal
// have to be rounded to floats

/*__device__ gmx_inline static void calc_exponentials_q_one(const real f, real &d, real &r, real &e)
{
  d = 1.0f/d;
  r = expf(r);
  e = f*r*d;
  }*/

//static const real sqrt_M_PI = sqrt(M_PI);
//static __constant__ real sqrt_M_PI_d;

/*__device__ gmx_inline static void calc_exponentials_lj_one(real &r, real &tmp2, real &d)
{
  d = 1.0f/d;
  r = exp(r);
  real mk = tmp2;
  tmp2 = sqrt_M_PI_d*mk*erfcf(mk);
  }*/

#define THREADS_PER_BLOCK (4 * warp_size)

template<
        const gmx_bool bEnerVir,
        const gmx_bool YZXOrdering
        //yupinov - now GPU solve works in a XYZ mode, while original solve worked in YZX order;
        // should be set to true when we do multi-rank GPU PME
        >
__global__ void pme_solve_kernel
(const int localCountMajor, const int localCountMiddle, const int localCountMinor,
 const int localOffsetMinor, const int localOffsetMajor, const int localOffsetMiddle,
 const int localSizeMinor, /*const int localSizeMajor,*/ const int localSizeMiddle,
 const int nMinor, const int nMajor, const int nMiddle,
 const real elfac, const real ewaldFactor,
 const real * __restrict__ BSplineModuleMinor,
 const real * __restrict__ BSplineModuleMajor,
 const real * __restrict__ BSplineModuleMiddle,
 float2 * __restrict__ grid,
 const real volume,
 #if !PME_EXTERN_CMEM
  const struct pme_gpu_recipbox_t RECIPBOX,
 #endif
 real * __restrict__ virialAndEnergy)
{
    // this is a PME solve kernel
    // each thread works on one cell of the Fourier space complex 3D grid (float2 * __restrict__ grid)
    // each block handles THREADS_PER_BLOCK cells - depending on the grid contiguous dimension size, that can range from a part of a single gridline to several complete gridlines
    // if we do cuFFT, the contiguous dimension (Z) gridlines are aligned by warp_size (but with respect to float, not float2, possibly?!)
    // also, the grid is in XYZ order as it was initially - single GPU only, no MPI decomposition, no transpose...
    // it should be possible to have multi-rank cuFFT - cuFFT should even be able to be linked instead of FFTW, and mimic its API
    // but it's outside the scope of my project ;-)
    // if we do FFTW, I forgot what happens with the size, probably, nothing and not very aligned
    // then the grid is in YZX order, so X is a contihuous dimension

    //const int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    const int threadLocalId = (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;
    //const int blockSize = (blockDim.x * blockDim.y * blockDim.z); // == cellsPerBlock
    const int blockSize = THREADS_PER_BLOCK;
    //const int threadId = blockId * blockSize + threadLocalId;


    int maxkMinor = (nMinor + 1) / 2;
    if (!YZXOrdering) //yupinov - don't really understand it
        maxkMinor = (nMinor + 2) / 2; // {[0 25]; -26} fixed, no maxkx required at all
    int maxkMajor = (nMajor + 1) / 2;
    //int maxkz = nz / 2 + 1;

    const int sizing = 7;

    real energy = 0.0f;
    real virxx = 0.0f, virxy = 0.0f, virxz = 0.0f, viryy = 0.0f, viryz = 0.0f, virzz = 0.0f;

    const int indexMinor = blockIdx.x * blockDim.x + threadIdx.x;
    const int indexMiddle = blockIdx.y * blockDim.y + threadIdx.y;
    const int indexMajor = blockIdx.z * blockDim.z + threadIdx.z;

    if ((indexMajor < localCountMajor) && (indexMiddle < localCountMiddle) && (indexMinor < localCountMinor))
    {
        const int kMajor = indexMajor + localOffsetMajor;
        const real mMajor = (kMajor < maxkMajor) ? kMajor : (kMajor - nMajor);

        const int kMiddle = indexMiddle + localOffsetMiddle;
        real mMiddle = kMiddle;

        if (!YZXOrdering)
            mMiddle = (kMiddle < (nMiddle + 1) / 2) ? kMiddle : (kMiddle - nMiddle);

        const real bMajorMiddle = real(M_PI) * volume * BSplineModuleMajor[kMajor] * BSplineModuleMiddle[kMiddle];

        // global complex grid pointer
        // the offset should be equal to threadId
        float2 *p0 = grid + (indexMajor * localSizeMiddle + indexMiddle) * localSizeMinor + indexMinor;


        /* We should skip the k-space point (0,0,0) */
        /* Note that since here x is the minor index, local_offset[XX]=0 */

        //yupinov fix dimension terminology

        int kMinor = localOffsetMinor + indexMinor;
        const gmx_bool notZeroPoint = (kMinor > 0 || kMajor > 0 || kMiddle > 0);
        real mMinor, mhxk, mhyk, mhzk, m2k;

        mMinor = kMinor < maxkMinor ? kMinor : (kMinor - nMinor);
        real mX, mY, mZ;
        if (YZXOrdering)
        {
            mX = mMinor;
            mY = mMajor;
            mZ = mMiddle;
        }
        else
        {
            mX = mMajor;
            mY = mMiddle;
            mZ = mMinor;
        }

        /* 0.5 correction for corner points of a Z dimension */
        real corner_fac = 1.0f;
        if (YZXOrdering)
        {
            if (kMiddle == 0 || kMiddle == (nMiddle + 1) / 2)
            {
                corner_fac = 0.5f;
            }
        }
        else
        {
            if (kMinor == 0 || kMinor == (nMinor + 1) / 2)
            {
                corner_fac = 0.5f;
            }
        }

        if (notZeroPoint) // this skips just one starting point in the whole grid on the rank 0
        {     
            mhxk      = mX * RECIPBOX.box[XX].x;
            mhyk      = mX * RECIPBOX.box[XX].y + mY * RECIPBOX.box[YY].y;
            mhzk      = mX * RECIPBOX.box[XX].z + mY * RECIPBOX.box[YY].z + mZ * RECIPBOX.box[ZZ].z;

            m2k       = mhxk * mhxk + mhyk * mhyk + mhzk * mhzk;
            real denom = m2k * bMajorMiddle * BSplineModuleMinor[kMinor];
            real tmp1  = -ewaldFactor * m2k;

            //calc_exponentials_q_one(elfac, denom, tmp1, eterm);
            denom = 1.0f / denom;
            tmp1 = expf(tmp1);
            real etermk = elfac * tmp1 * denom;

            float2 gridValue = *p0;
            gridValue.x *= etermk;
            gridValue.y *= etermk;
            *p0 = gridValue;

            if (bEnerVir)
            {
                real tmp1k = 2.0f * (gridValue.x * gridValue.x + gridValue.y * gridValue.y) / etermk;

                real vfactor = (ewaldFactor + 1.0f / m2k) * 2.0f;
                real ets2 = corner_fac * tmp1k;
                energy = ets2;

                real ets2vf  = ets2 * vfactor;

                virxx   = ets2vf * mhxk * mhxk - ets2;
                virxy   = ets2vf * mhxk * mhyk;
                virxz   = ets2vf * mhxk * mhzk;
                viryy   = ets2vf * mhyk * mhyk - ets2;
                viryz   = ets2vf * mhyk * mhzk;
                virzz   = ets2vf * mhzk * mhzk - ets2;
            }
        }
    }

    if (bEnerVir)
    {
        // reduction goes here

#if (GMX_PTX_ARCH >= 300)
        // there should be a shuffle reduction here
        /*
        if (!(blockSize & (blockSize - 1))) // only for orders of power of 2
        {

        }
        else
        */
#endif
        {
            __shared__ real virialAndEnergyShared[sizing * blockSize];
            // 3.5k smem per block - a serious limiter!

            // 7-thread reduction in shared memory inspired by reduce_force_j_generic
            virialAndEnergyShared[threadLocalId + 0 * blockSize] = virxx;
            virialAndEnergyShared[threadLocalId + 1 * blockSize] = viryy;
            virialAndEnergyShared[threadLocalId + 2 * blockSize] = virzz;
            virialAndEnergyShared[threadLocalId + 3 * blockSize] = virxy;
            virialAndEnergyShared[threadLocalId + 4 * blockSize] = virxz;
            virialAndEnergyShared[threadLocalId + 5 * blockSize] = viryz;
            virialAndEnergyShared[threadLocalId + 6 * blockSize] = energy;

            __syncthreads();

            // reduce every component to fit into warp_size
            for (int s = blockSize >> 1; s >= warp_size; s >>= 1)
            {
#pragma unroll
                for (int i = 0; i < sizing; i++)
                {
                    if (threadLocalId < s) //again, split per threads ?
                        virialAndEnergyShared[i * blockSize + threadLocalId] += virialAndEnergyShared[i * blockSize + threadLocalId + s];
                }
                __syncthreads();
            }

            const int threadsPerComponent = warp_size / sizing; // this is also the stride
            if (threadLocalId < sizing * threadsPerComponent)
            {
                const int componentIndex = threadLocalId / threadsPerComponent;
                const int threadComponentOffset = threadLocalId - componentIndex * threadsPerComponent;

                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < sizing; j++)
                {
                    sum += virialAndEnergyShared[componentIndex * blockSize + j * threadsPerComponent + threadComponentOffset];
                }
                // write to global memory
                atomicAdd(virialAndEnergy + componentIndex, sum);
            }
            /*
            // a naive shared mem reduction
            virialAndEnergyShared[sizing * threadLocalId + 0] = virxx;
            virialAndEnergyShared[sizing * threadLocalId + 1] = viryy;
            virialAndEnergyShared[sizing * threadLocalId + 2] = virzz;
            virialAndEnergyShared[sizing * threadLocalId + 3] = virxy;
            virialAndEnergyShared[sizing * threadLocalId + 4] = virxz;
            virialAndEnergyShared[sizing * threadLocalId + 5] = viryz;
            virialAndEnergyShared[sizing * threadLocalId + 6] = energy;

            __syncthreads();
#pragma unroll
            for (unsigned int stride = 1; stride < blockSize; stride <<= 1)
            {
                if ((threadLocalId % (stride << 1) == 0))
                {
#pragma unroll
                    for (int i = 0; i < sizing; i++)
                        virialAndEnergyShared[sizing * threadLocalId + i] += virialAndEnergyShared[sizing * (threadLocalId + stride) + i];
                }
                __syncthreads();
            }
            if (threadLocalId < sizing)
                atomicAdd(virialAndEnergy + threadLocalId, virialAndEnergyShared[threadLocalId]);
            */
        }
    }
}

void solve_pme_gpu(struct gmx_pme_t *pme, t_complex *grid,
                       real ewaldcoeff, real vol,
                       gmx_bool bEnerVir, int thread)
{
    const gmx_bool YZXOrdering = !pme->bGPUFFT;
    /* do recip sum over local cells in grid */

    if (thread != 0) //yupinov check everywhere inside!
        return;

    cudaStream_t s = pme->gpu->pmeStream;

    ivec complex_order, local_ndata, local_offset, local_size;
    /* Dimensions should be identical for A/B grid, so we just use A here */
    gmx_parallel_3dfft_complex_limits_wrapper(pme, PME_GRID_QA,//pme->pfft_setup_gpu[PME_GRID_QA],
                                      complex_order,
                                      local_ndata,
                                      local_offset,
                                      local_size);
    //yupinov replace with gmx_parallel_3dfft_complex_limits_gpu



    /* true: y major, z middle, x minor or continuous - the CPU FFT way */
    /* false: x major, y middle, z minor - the single rank GPU cuFFT way */

    //yupinov fix YZXOrdering pecularities in solve
    const int minorDim = !YZXOrdering ? ZZ : XX;
    const int middleDim = !YZXOrdering ? YY : ZZ;
    const int majorDim = !YZXOrdering ? XX : YY;
    const int nMinor = !YZXOrdering ? pme->nkz : pme->nkx;
    const int nMajor = !YZXOrdering ? pme->nkx : pme->nky;
    const int nMiddle = !YZXOrdering ? pme->nky : pme->nkz;

    const real elfac = ONE_4PI_EPS0 / pme->epsilon_r; // make it a constant as well

    //yupinov align minor dimension with cachelines!

    //const int n = local_ndata[majorDim] * local_ndata[middleDim] * local_ndata[minorDim];
    const int grid_n = local_size[majorDim] * local_size[middleDim] * local_size[minorDim];
    const int grid_size = grid_n * sizeof(float2);

    real *bspModMinor_d = (real *)PMEMemoryFetchAndCopy(PME_ID_BSP_MOD_MINOR, thread, pme->bsp_mod[minorDim], nMinor * sizeof(real), ML_DEVICE, s);
    real *bspModMajor_d = (real *)PMEMemoryFetchAndCopy(PME_ID_BSP_MOD_MAJOR, thread, pme->bsp_mod[majorDim], nMajor * sizeof(real), ML_DEVICE, s);
    real *bspModMiddle_d = (real *)PMEMemoryFetchAndCopy(PME_ID_BSP_MOD_MIDDLE, thread, pme->bsp_mod[middleDim], nMiddle * sizeof(real), ML_DEVICE, s);

    float2 *grid_d = (float2 *)PMEMemoryFetch(PME_ID_COMPLEX_GRID, thread, grid_size, ML_DEVICE); //yupinov no need for special function
    if (!pme->gpu->keepGPUDataBetweenR2CAndSolve)
        PMEMemoryCopy(grid_d, grid, grid_size, ML_DEVICE, s);

    const real ewaldFactor = (M_PI * M_PI) / (ewaldcoeff * ewaldcoeff);

    // Z-dimension is too small in CUDA limitations (64 on CC30?), so instead of major-middle-minor sizing we do minor-middle-major
    const int blockSize = THREADS_PER_BLOCK;
    /*
    dim3 blocks((local_ndata[minorDim] + blockSize - 1) / blockSize, local_ndata[middleDim], local_ndata[majorDim]);
    dim3 threads(blockSize, 1, 1);
    */
    //yupinov check ALIGNMENT with CPU/GPU FFT grid sizes!
    const int gridLineSize = local_size[minorDim];
    const int gridLinesPerBlock = max(blockSize / gridLineSize, 1);

    //yupinov check all block dimensions for rounding
    dim3 blocks((gridLineSize + blockSize - 1) / blockSize,  // rounded up blocks per grid line
                (local_ndata[middleDim] + gridLinesPerBlock - 1) / gridLinesPerBlock, // rounded up middle dimension block number
                local_ndata[majorDim]);
    // integer number of blocks is working on integer number of gridlines
    dim3 threads(gridLineSize, gridLinesPerBlock, 1);

    const int nReduced = 1;
    const int energyAndVirialSize = nReduced * (1 + 6) * sizeof(real);
    real *energyAndVirial_d = (real *)PMEMemoryFetch(PME_ID_ENERGY_AND_VIRIAL, thread, energyAndVirialSize, ML_DEVICE);
    cudaError_t stat = cudaMemsetAsync(energyAndVirial_d, 0, energyAndVirialSize, s);
    CU_RET_ERR(stat, "PME solve cudaMemsetAsync");

    pme_gpu_timing_start(pme, ewcsPME_SOLVE);

    if (YZXOrdering)
    {
        if (bEnerVir)
            pme_solve_kernel<TRUE, TRUE> <<<blocks, threads, 0, s>>>
              (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim],
               local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
               local_size[minorDim], /*local_size[majorDim],*/ local_size[middleDim],
               nMinor, nMajor, nMiddle,
               elfac, ewaldFactor,
               bspModMinor_d, bspModMajor_d, bspModMiddle_d,
               grid_d, vol,
#if !PME_EXTERN_CMEM
               pme->gpu->recipbox,
#endif
               energyAndVirial_d);
        else
            pme_solve_kernel<FALSE, TRUE> <<<blocks, threads, 0, s>>>
              (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim ],
               local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
               local_size[minorDim], /*local_size[majorDim],*/local_size[middleDim],
               nMinor, nMajor, nMiddle,
               elfac, ewaldFactor,
               bspModMinor_d, bspModMajor_d, bspModMiddle_d,
               grid_d, vol,
#if !PME_EXTERN_CMEM
               pme->gpu->recipbox,
#endif
               energyAndVirial_d);
    }
    else
    {
        if (bEnerVir)
            pme_solve_kernel<TRUE, FALSE> <<<blocks, threads, 0, s>>>
              (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim],
               local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
               local_size[minorDim], /*local_size[majorDim],*/ local_size[middleDim],
               nMinor, nMajor, nMiddle,
               elfac, ewaldFactor,
               bspModMinor_d, bspModMajor_d, bspModMiddle_d,
               grid_d, vol,
#if !PME_EXTERN_CMEM
                pme->gpu->recipbox,
#endif
               energyAndVirial_d);
        else
            pme_solve_kernel<FALSE, FALSE> <<<blocks, threads, 0, s>>>
              (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim ],
               local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
               local_size[minorDim], /*local_size[majorDim],*/local_size[middleDim],
               nMinor, nMajor, nMiddle,
               elfac, ewaldFactor,
               bspModMinor_d, bspModMajor_d, bspModMiddle_d,
               grid_d, vol,
#if !PME_EXTERN_CMEM
               pme->gpu->recipbox,
#endif
               energyAndVirial_d);
    }
    CU_LAUNCH_ERR("pme_solve_kernel");

    pme_gpu_timing_stop(pme, ewcsPME_SOLVE);

    if (!pme->gpu->keepGPUDataBetweenSolveAndC2R)
    {
        PMEMemoryCopy(grid, grid_d, grid_size, ML_HOST, s);
    }

    if (bEnerVir)
    {
        (real *)PMEMemoryFetchAndCopy(PME_ID_ENERGY_AND_VIRIAL, thread, energyAndVirial_d, energyAndVirialSize, ML_HOST, s);
        stat = cudaEventRecord(pme->gpu->syncEnerVirH2D, s);
        CU_RET_ERR(stat, "PME solve energy/virial sync fail");
    }

    /* Return the loop count */
    //return local_ndata[YY]*local_ndata[XX]; //yupinov why
}

void pme_gpu_get_energy_virial(gmx_pme_t *pme)
{
    const int thread = 0;

    cudaStream_t s = pme->gpu->pmeStream;

    struct pme_solve_work_t *work = &pme->solve_work[thread];
    real *work_energy_q = &(work->energy_q);
    matrix &work_vir_q = work->vir_q;

    int energyAndVirialSize = PMEGetAllocatedSize(PME_ID_ENERGY_AND_VIRIAL, thread, ML_DEVICE);
    assert(energyAndVirialSize > 0);
    int nReduced = energyAndVirialSize / (1 + 6) / sizeof(real);
    // will be 1, actually

    cudaError_t stat = cudaStreamWaitEvent(s, pme->gpu->syncEnerVirH2D, 0);
    CU_RET_ERR(stat, "error while waiting for PME solve");
    real *energyAndVirial_h = (real *)PMEMemoryFetch(PME_ID_ENERGY_AND_VIRIAL, thread, energyAndVirialSize, ML_HOST);
    real energy = 0.0;
    real virxx = 0.0, virxy = 0.0, virxz = 0.0, viryy = 0.0, viryz = 0.0, virzz = 0.0;
    for (int i = 0, j = 0; i < nReduced; ++i)
    {
        virxx += energyAndVirial_h[j++];
        viryy += energyAndVirial_h[j++];
        virzz += energyAndVirial_h[j++];
        virxy += energyAndVirial_h[j++];
        virxz += energyAndVirial_h[j++];
        viryz += energyAndVirial_h[j++];
        energy += energyAndVirial_h[j++];
    }

    work_vir_q[XX][XX] = 0.25 * virxx;
    work_vir_q[YY][YY] = 0.25 * viryy;
    work_vir_q[ZZ][ZZ] = 0.25 * virzz;
    work_vir_q[XX][YY] = work_vir_q[YY][XX] = 0.25 * virxy;
    work_vir_q[XX][ZZ] = work_vir_q[ZZ][XX] = 0.25 * virxz;
    work_vir_q[YY][ZZ] = work_vir_q[ZZ][YY] = 0.25 * viryz;

    /* This energy should be corrected for a charged system */
    *work_energy_q = 0.5 * energy;
}


__global__ void solve_pme_lj_yzx_iyz_loop_kernel
(int iyz0, int iyz1, int local_ndata_ZZ, int local_ndata_XX,
 int local_offset_XX, int local_offset_YY, int local_offset_ZZ,
 int local_size_XX, int local_size_YY, int local_size_ZZ,
 int nx, int ny, int nz,
 real rxx, real ryx, real ryy, real rzx, real rzy, real rzz,
 //real ,
 //splinevec pme_bsp_mod,
 real *pme_bsp_mod_XX, real *pme_bsp_mod_YY, real *pme_bsp_mod_ZZ,
 float2 *grid_v, gmx_bool bLB,
 real ewaldcoeff, real vol,
 gmx_bool bEnerVir,
 real *energy_v, real *virial_v) {

    const int grid_size = local_size_YY * local_size_ZZ * local_size_XX;
    const real factor = M_PI*M_PI/(ewaldcoeff*ewaldcoeff);

    int maxkx = (nx+1)/2;
    int maxky = (ny+1)/2;
    //int maxkz = nz/2+1;
    //(void) maxkz; // unused


    real energy = 0.0f;
    real virxx = 0.0f, virxy = 0.0f, virxz = 0.0f, viryy = 0.0f, viryz = 0.0f, virzz = 0.0f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int iyz = iyz0 + i;
    if (iyz < iyz1)
    {
        int iy = iyz/local_ndata_ZZ;
        int iz = iyz - iy*local_ndata_ZZ;

        int ky = iy + local_offset_YY;
        real my;

        if (ky < maxky)
        {
            my = ky;
        }
        else
        {
            my = (ky - ny);
        }

        real by = 3.0f * vol * pme_bsp_mod_YY[ky]
                / (M_PI*sqrt(M_PI)*ewaldcoeff*ewaldcoeff*ewaldcoeff); //yupinov double!

        int kz = iz + local_offset_ZZ;

        real mz = kz;

        real bz = pme_bsp_mod_ZZ[kz];

        /* 0.5 correction for corner points */
        real corner_fac = 1.0f;
        if (kz == 0 || kz == (nz+1)/2)
        {
            corner_fac = 0.5f;
        }

        int kxstart = local_offset_XX;
        int kxend   = local_offset_XX + local_ndata_XX;

        real mx, mhxk, mhyk, mhzk, m2k;

        if (bEnerVir)
        {
            float2 *p0 = grid_v/*[0]*/ + iy*local_size_ZZ*local_size_XX + iz*local_size_XX;
            /* More expensive inner loop, especially because of the
             * storage of the mh elements in array's.  Because x is the
             * minor grid index, all mh elements depend on kx for
             * triclinic unit cells.
             */

            // /* Two explicit loops to avoid a conditional inside the loop */
            // NOTE: on gpu, keep the conditional. shouldn't be too bad?
            for (int kx = kxstart; kx < kxend; kx++, p0++)
            {
                mx = kx < maxkx ? kx : (kx - nx);

                mhxk      = mx * rxx;
                mhyk      = mx * ryx + my * ryy;
                mhzk      = mx * rzx + my * rzy + mz * rzz;
                m2k       = mhxk*mhxk + mhyk*mhyk + mhzk*mhzk;
                real denomk = bz*by*pme_bsp_mod_XX[kx];
                real tmp1k  = -factor*m2k;
                real tmp2k  = sqrt(factor*m2k);

                //calc_exponentials_lj_one(tmp1k, tmp2k, denomk); // r tmp2 d
                denomk = 1.0f / denomk;
                tmp1k = exp(tmp1k);
                real mk = tmp2k;
                tmp2k = SQRT_M_PI * mk * erfcf(mk);

                m2k   = factor*m2k;
                real eterm = -((1.0f - 2.0f * m2k) * tmp1k
                               + 2.0f * m2k * tmp2k);
                real vterm    = 3.0f * (-tmp1k + tmp2k);
                tmp1k = eterm*denomk;
                tmp2k = vterm*denomk;

                if (!bLB)
                {
                    real d1      = p0->x;
                    real d2      = p0->y;

                    eterm   = tmp1k;
                    vterm   = tmp2k;
                    p0->x  = d1*eterm;
                    p0->y  = d2*eterm;

                    real struct2 = 2.0f * (d1 * d1 + d2 * d2);

                    tmp1k = eterm*struct2;
                    tmp2k = vterm*struct2;
                }
                else
                {
                    //real *struct2 = denom;
                    real  str2;

                    real struct2k = 0.0f;

                    /* Due to symmetry we only need to calculate 4 of the 7 terms */
                    for (int ig = 0; ig <= 3; ++ig)
                    {
                        //t_complex *p0, *p1;
                        real       scale;

                        float2 *p0k    = grid_v/*[ig]*/ + ig*grid_size + iy*local_size_ZZ*local_size_XX + iz*local_size_XX + (kx - kxstart);
                        float2 *p1k    = grid_v/*[6-ig]*/ + (6-ig)*grid_size + iy*local_size_ZZ*local_size_XX + iz*local_size_XX + (kx - kxstart);
                        scale = 2.0f * lb_scale_factor_symm_gpu[ig];
                        struct2k += scale*(p0k->x*p1k->x + p0k->y*p1k->y);
                    }
                    for (int ig = 0; ig <= 6; ++ig)
                    {
                        //t_complex *p0;

                        float2 *p0k = grid_v/*[ig]*/ + ig*grid_size + iy*local_size_ZZ*local_size_XX + iz*local_size_XX + (kx - kxstart);

                        real d1     = p0k->x;
                        real d2     = p0k->y;

                        eterm  = tmp1k;
                        p0k->x = d1*eterm;
                        p0k->y = d2*eterm;
                    }

                    eterm    = tmp1k;
                    vterm    = tmp2k;
                    str2     = struct2k;
                    tmp1k = eterm*str2;
                    tmp2k = vterm*str2;
                }

                real ets2     = corner_fac*tmp1k;
                vterm    = 2.0f * factor*tmp2k;
                energy  += ets2;
                real ets2vf   = corner_fac*vterm;
                virxx   += ets2vf*mhxk*mhxk - ets2;
                virxy   += ets2vf*mhxk*mhyk;
                virxz   += ets2vf*mhxk*mhzk;
                viryy   += ets2vf*mhyk*mhyk - ets2;
                viryz   += ets2vf*mhyk*mhzk;
                virzz   += ets2vf*mhzk*mhzk - ets2;
            }
        }
        else
        {
            /* We don't need to calculate the energy and the virial.
             *  In this case the triclinic overhead is small.
             */

            /* Two explicit loops to avoid a conditional inside the loop */
            // NOTE: on gpu, keep the conditional. shouldn't be too bad?
            for (int kx = kxstart; kx < kxend; kx++)
            {
                mx = kx < maxkx ? kx : (kx - nx);

                mhxk      = mx * rxx;
                mhyk      = mx * ryx + my * ryy;
                mhzk      = mx * rzx + my * rzy + mz * rzz;
                real m2k       = mhxk*mhxk + mhyk*mhyk + mhzk*mhzk;
                real denomk = bz*by*pme_bsp_mod_XX[kx];
                real tmp1k  = -factor*m2k;
                real tmp2k  = sqrt(factor*m2k);

                //calc_exponentials_lj_one(tmp1k, tmp2k, denomk); // r tmp2 d
                denomk = 1.0f / denomk;
                tmp1k = exp(tmp1k);
                real mk = tmp2k;
                tmp2k = SQRT_M_PI * mk * erfcf(mk); //yupinov std::erfc? gmx_erfc?

                m2k    = factor*m2k;
                real eterm  = -((1.0f - 2.0f*m2k)*tmp1k
                                + 2.0f*m2k*tmp2k);
                tmp1k = eterm*denomk;

                int gcount = (bLB ? 7 : 1);
                for (int ig = 0; ig < gcount; ++ig)
                {
                    //t_complex *p0;

                    float2 *p0k = grid_v/*[ig]*/ + ig*grid_size + iy*local_size_ZZ*local_size_XX + iz*local_size_XX + (kx - kxstart);

                    real d1      = p0k->x;
                    real d2      = p0k->y;

                    eterm   = tmp1k;

                    p0k->x  = d1*eterm;
                    p0k->y  = d2*eterm;
                }
            }
        }
        energy_v[i] = energy;
        virial_v[0] = virxx;
        virial_v[1] = viryy;
        virial_v[2] = virzz;
        virial_v[3] = virxy;
        virial_v[4] = virxz;
        virial_v[5] = viryz;
    }
}



int solve_pme_lj_yzx_gpu(int nx, int ny, int nz,
			 ivec complex_order, ivec local_ndata, ivec local_offset, ivec local_size,
			 real rxx, real ryx, real ryy, real rzx, real rzy, real rzz,
			 //real *mhx, real *mhy, real *mhz, real *m2, real *denom, real *tmp1, real *tmp2,
			 splinevec pme_bsp_mod,
			 matrix work_vir_lj, real *work_energy_lj,
			 t_complex **grid, gmx_bool bLB,
			 real ewaldcoeff, real vol,
             gmx_bool bEnerVir, gmx_pme_t *pme, int nthread, int thread)
{
    cudaStream_t s = pme->gpu->pmeStream;
    /* do recip sum over local cells in grid */
    /* y major, z middle, x minor or continuous */
    //int     ig, gcount;
    //int     kx, ky, kz, maxkx, maxky, maxkz;
    int     /*iy,*/ iyz0, iyz1; //, iyz, iz, kxstart, kxend;
    //real    mx, my, mz;
    //real    factor = M_PI*M_PI/(ewaldcoeff*ewaldcoeff);
    //real    ets2, ets2vf;
    //real    eterm, vterm, d1, d2;
    real energy = 0.0;
    //real    by, bz;
    real    virxx = 0.0, virxy = 0.0, virxz = 0.0, viryy = 0.0, viryz = 0.0, virzz = 0.0;
    //real    mhxk, mhyk, mhzk, m2k;
    //real    mk;
    //real    corner_fac;

    /* Dimensions should be identical for A/B grid, so we just use A here */
    /* Dimensions are passed in. TODO: call elsewhere?
    gmx_parallel_3dfft_complex_limits(pme->pfft_setup[PME_GRID_C6A],
                                      complex_order,
                                      local_ndata,
                                      local_offset,
                                      local_size);
    gmx_parallel_3dfft_complex_limits_gpu(pme->pfft_setup_gpu[PME_GRID_C6A],
                                      complex_order,
                                      local_ndata,
                                      local_offset,
                                      local_size);
    */

    iyz0 = local_ndata[YY]*local_ndata[ZZ]* thread   /nthread;
    iyz1 = local_ndata[YY]*local_ndata[ZZ]*(thread+1)/nthread;

    const int block_size = warp_size;
    int n = iyz1 - iyz0;
    int n_blocks = (n + block_size - 1) / block_size;

#define MAGIC_GRID_NUMBER 6
    //yupinov

    int grid_n = local_size[YY] * local_size[ZZ] * local_size[XX];
    int grid_size = grid_n * sizeof(t_complex);
    float2 *grid_d = (float2 *)PMEMemoryFetch(PME_ID_COMPLEX_GRID, thread, grid_size * MAGIC_GRID_NUMBER, ML_DEVICE); //6 grids!
    real *pme_bsp_mod_x_d = (real *)PMEMemoryFetchAndCopy(PME_ID_BSP_MOD_MINOR, thread, pme_bsp_mod[XX], nx * sizeof(real), ML_DEVICE, s);
    real *pme_bsp_mod_y_d = (real *)PMEMemoryFetchAndCopy(PME_ID_BSP_MOD_MAJOR, thread, pme_bsp_mod[YY], ny * sizeof(real), ML_DEVICE, s);
    real *pme_bsp_mod_z_d = (real *)PMEMemoryFetchAndCopy(PME_ID_BSP_MOD_MIDDLE, thread, pme_bsp_mod[ZZ], nz * sizeof(real), ML_DEVICE, s);
    int energy_size = n * sizeof(real);
    int virial_size = 6 * n * sizeof(real);
    real *energy_d = (real *)PMEMemoryFetch(PME_ID_ENERGY, thread, energy_size, ML_DEVICE);
    real *virial_d = (real *)PMEMemoryFetch(PME_ID_VIRIAL, thread, virial_size, ML_DEVICE);
    for (int ig = 0; ig < MAGIC_GRID_NUMBER; ++ig)
        PMEMemoryCopy(grid_d + ig * grid_n, grid[ig], grid_size, ML_DEVICE, s);

    pme_gpu_timing_start(pme, ewcsPME_SOLVE);

    solve_pme_lj_yzx_iyz_loop_kernel<<<n_blocks, block_size, 0, s>>>
      (iyz0, iyz1, local_ndata[ZZ], local_ndata[XX],
       local_offset[XX], local_offset[YY], local_offset[ZZ],
       local_size[XX], local_size[YY], local_size[ZZ],
       nx, ny, nz, rxx, ryx, ryy, rzx, rzy, rzz,
       //,
       //pme_bsp_mod,
       pme_bsp_mod_x_d, pme_bsp_mod_y_d, pme_bsp_mod_z_d,
       grid_d, bLB, ewaldcoeff, vol, bEnerVir,
       energy_d, virial_d);
    CU_LAUNCH_ERR("solve_pme_lj_yzx_iyz_loop_kernel");

    pme_gpu_timing_stop(pme, ewcsPME_SOLVE);

    for (int ig = 0; ig < MAGIC_GRID_NUMBER; ++ig)
        PMEMemoryCopy(grid[ig], grid_d + ig * grid_n, grid_size, ML_HOST, s);

    if (bEnerVir)
    {
        real *energy_h = (real *)PMEMemoryFetchAndCopy(PME_ID_ENERGY, thread, energy_d, energy_size, ML_HOST, s);
        real *virial_h = (real *)PMEMemoryFetchAndCopy(PME_ID_VIRIAL, thread, virial_d, virial_size, ML_HOST, s);
        //yupinov - workaround for a zero point - do in kernel?
        memset(energy_h, 0, sizeof(real));
        memset(virial_h, 0, 6 * sizeof(real));

        for (int i = 0, j = 0; i < n; ++i)
        {
            energy += energy_h[i];
            virxx += virial_h[j++];
            viryy += virial_h[j++];
            virzz += virial_h[j++];
            virxy += virial_h[j++];
            virxz += virial_h[j++];
            viryz += virial_h[j++];
        }

        work_vir_lj[XX][XX] = 0.25 * virxx;
        work_vir_lj[YY][YY] = 0.25 * viryy;
        work_vir_lj[ZZ][ZZ] = 0.25 * virzz;
        work_vir_lj[XX][YY] = work_vir_lj[YY][XX] = 0.25 * virxy;
        work_vir_lj[XX][ZZ] = work_vir_lj[ZZ][XX] = 0.25 * virxz;
        work_vir_lj[YY][ZZ] = work_vir_lj[ZZ][YY] = 0.25 * viryz;

        /* This energy should be corrected for a charged system */
        *work_energy_lj = 0.5 * energy;
    }
    /* Return the loop count */
    return local_ndata[YY]*local_ndata[XX];
}



