#include "pme.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

#include "gromacs/utility/gmxassert.h"
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
// static const __constant__ real lb_scale_factor_symm_gpu[] = { 2.0/64, 12.0/64, 30.0/64, 20.0/64 };
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

void pme_gpu_alloc_energy_virial(gmx_pme_t *pme, const int grid_index)
{
    pme->gpu->energyAndVirialSize = 7 * sizeof(real); // 6 virial components + energy
    pme->gpu->energyAndVirial = (real *)PMEMemoryFetch(PME_ID_ENERGY_AND_VIRIAL, pme->gpu->energyAndVirialSize, ML_DEVICE);
}

void pme_gpu_clear_energy_virial(gmx_pme_t *pme, const int grid_index)
{
    cudaError_t stat = cudaMemsetAsync(pme->gpu->energyAndVirial, 0, pme->gpu->energyAndVirialSize, pme->gpu->pmeStream);
    CU_RET_ERR(stat, "PME solve energies/virial cudaMemsetAsync");
}

void pme_gpu_copy_bspline_moduli(gmx_pme_t *pme)
{
    //yupinov make it textures

    for (int i = 0; i < DIM; i++)
    {
        int n;
        PMEDataID id;
        switch (i)
        {
            case XX:
            n = pme->nkx;
            id = PME_ID_BSP_MOD_XX;
            break;

            case YY:
            n = pme->nky;
            id = PME_ID_BSP_MOD_YY;
            break;

            case ZZ:
            n = pme->nkz;
            id = PME_ID_BSP_MOD_ZZ;
            break;
        }
        int modSize = n * sizeof(real);
        real *bspMod_h = (real *)PMEMemoryFetch(id, modSize, ML_HOST);
        memcpy(bspMod_h, pme->bsp_mod[i], modSize);
        real *bspMod_d = (real *)PMEMemoryFetch(id, modSize, ML_DEVICE);
        cu_copy_H2D_async(bspMod_d, bspMod_h, modSize, pme->gpu->pmeStream);
    }
}


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


    int maxkMajor = (nMajor + 1) / 2; //X or Y
    int maxkMiddle = (nMiddle + 1) / 2; //Y OR Z => only check for !YZX
    int maxkMinor = (nMinor + 1) / 2; //Z or X => only check for YZX

    const int sizing = 7;

    real energy = 0.0f;
    real virxx = 0.0f, virxy = 0.0f, virxz = 0.0f, viryy = 0.0f, viryz = 0.0f, virzz = 0.0f;

    const int indexMinor = blockIdx.x * blockDim.x + threadIdx.x;
    const int indexMiddle = blockIdx.y * blockDim.y + threadIdx.y;
    const int indexMajor = blockIdx.z * blockDim.z + threadIdx.z;

    if ((indexMajor < localCountMajor) && (indexMiddle < localCountMiddle) && (indexMinor < localCountMinor))
    {
        const int kMajor = indexMajor + localOffsetMajor;
        // check X in XYZ or Y in YZX
        const real mMajor = (kMajor < maxkMajor) ? kMajor : (kMajor - nMajor);

        const int kMiddle = indexMiddle + localOffsetMiddle;
        real mMiddle = kMiddle;

        // check Y in XYZ
        if (!YZXOrdering)
            mMiddle = (kMiddle < maxkMiddle) ? kMiddle : (kMiddle - nMiddle);



        const real bMajorMiddle = real(M_PI) * volume * BSplineModuleMajor[kMajor] * BSplineModuleMiddle[kMiddle];

        // global complex grid pointer
        // the offset should be equal to threadId
        float2 *p0 = grid + (indexMajor * localSizeMiddle + indexMiddle) * localSizeMinor + indexMinor;


        /* We should skip the k-space point (0,0,0) */
        /* Note that since here x is the minor index, local_offset[XX]=0 */

        const int kMinor = localOffsetMinor + indexMinor;
        const gmx_bool notZeroPoint = (kMinor > 0 || kMajor > 0 || kMiddle > 0);
        real mMinor = kMinor, mhxk, mhyk, mhzk, m2k;

        // check X in YZX
        if (YZXOrdering)
            mMinor = (kMinor < maxkMinor) ? kMinor : (kMinor - nMinor);
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

        const gmx_bool debugPrint = false;//(mX== 7) && (mY == 7) && (mZ == 7);

        /* 0.5 correction for corner points of a Z dimension */
        real corner_fac = 1.0f;
        if (YZXOrdering)
        {
            if (kMiddle == 0 || kMiddle == maxkMiddle)
            {
                corner_fac = 0.5f;
            }
        }
        else
        {
            if (kMinor == 0 || kMinor == maxkMinor)
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
            if (debugPrint)
                printf("grid %g %g\n", gridValue.x, gridValue.y);

            gridValue.x *= etermk;
            gridValue.y *= etermk;
            *p0 = gridValue;

            if (debugPrint)
                printf("grid %g %g\n", gridValue.x, gridValue.y);

            if (bEnerVir)
            {
                real tmp1k = 2.0f * (gridValue.x * gridValue.x + gridValue.y * gridValue.y) / etermk;

                real vfactor = (ewaldFactor + 1.0f / m2k) * 2.0f;
                real ets2 = corner_fac * tmp1k;
                energy = ets2;
                if (debugPrint)// ||isnan(energy))
                    printf("energy %g %g %g %g\n", energy, mX, mY, mZ);
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
                       gmx_bool bEnerVir)
{
    const gmx_bool YZXOrdering = !pme->bGPUFFT;
    /* do recip sum over local cells in grid */

    cudaStream_t s = pme->gpu->pmeStream;

    ivec complex_order, local_ndata, local_offset, local_size;
    /* Dimensions should be identical for A/B grid, so we just use A here */
    gmx_parallel_3dfft_complex_limits_wrapper(pme, PME_GRID_QA,//pme->pfft_setup_gpu[PME_GRID_QA],
                                      complex_order,
                                      local_ndata,
                                      local_offset,
                                      local_size);
    //here we have correct complex ndata and sizes for CPU/GPU FFT


    /* true: y major, z middle, x minor or continuous - the CPU FFT way */
    /* false: x major, y middle, z minor - the single rank GPU cuFFT way */

    //yupinov fix YZXOrdering pecularities in solve
    const int minorDim = !YZXOrdering ? ZZ : XX;
    const int middleDim = !YZXOrdering ? YY : ZZ;
    const int majorDim = !YZXOrdering ? XX : YY;

    const int nMinor = !YZXOrdering ? pme->nkz : pme->nkx;
    const int nMajor = !YZXOrdering ? pme->nkx : pme->nky;
    const int nMiddle = !YZXOrdering ? pme->nky : pme->nkz;

    /*
    const int nMinor =  local_ndata[minorDim]; //!YZXOrdering ? pme->nkz : pme->nkx;
    const int nMajor = local_ndata[majorDim];
    const int nMiddle = local_ndata[middleDim]; //these are basic sizes, so what
    */
    const real *bspModMinor_d = (real *)PMEMemoryFetch(!YZXOrdering ? PME_ID_BSP_MOD_ZZ : PME_ID_BSP_MOD_XX, 0, ML_DEVICE);
    const real *bspModMiddle_d = (real *)PMEMemoryFetch(!YZXOrdering ? PME_ID_BSP_MOD_YY : PME_ID_BSP_MOD_ZZ, 0, ML_DEVICE);
    const real *bspModMajor_d = (real *)PMEMemoryFetch(!YZXOrdering ? PME_ID_BSP_MOD_XX : PME_ID_BSP_MOD_YY, 0, ML_DEVICE);

    const real elfac = ONE_4PI_EPS0 / pme->epsilon_r; // make it a constant as well

    //yupinov align minor dimension with cachelines!

    //const int n = local_ndata[majorDim] * local_ndata[middleDim] * local_ndata[minorDim];
    const int grid_n = local_size[majorDim] * local_size[middleDim] * local_size[minorDim];
    const int grid_size = grid_n * sizeof(float2);

    float2 *grid_d = (float2 *)PMEMemoryFetch(PME_ID_COMPLEX_GRID, grid_size, ML_DEVICE); //yupinov no need for special function
    if (!pme->gpu->keepGPUDataBetweenR2CAndSolve)
    {
        cu_copy_H2D_async(grid_d, grid, grid_size, s); //sync...
    }
    const real ewaldFactor = (M_PI * M_PI) / (ewaldcoeff * ewaldcoeff);

    // Z-dimension is too small in CUDA limitations (64 on CC30?), so instead of major-middle-minor sizing we do minor-middle-major
    const int blockSize = THREADS_PER_BLOCK;

    const int gridLineSize = local_size[minorDim];
    const int gridLinesPerBlock = max(blockSize / gridLineSize, 1);

    dim3 blocks((gridLineSize + blockSize - 1) / blockSize,  // rounded up blocks per grid line
                (local_ndata[middleDim] + gridLinesPerBlock - 1) / gridLinesPerBlock, // rounded up middle dimension block number
                local_ndata[majorDim]);
    // integer number of blocks is working on integer number of gridlines
    dim3 threads(gridLineSize, gridLinesPerBlock, 1);

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
               pme->gpu->energyAndVirial);
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
               pme->gpu->energyAndVirial);
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
               pme->gpu->energyAndVirial);
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
               pme->gpu->energyAndVirial);
    }
    CU_LAUNCH_ERR("pme_solve_kernel");

    pme_gpu_timing_stop(pme, ewcsPME_SOLVE);

    if (!pme->gpu->keepGPUDataBetweenSolveAndC2R)
    {
        cu_copy_D2H_async(grid, grid_d, grid_size, s);
    }

    if (bEnerVir)
    {
        real *energyAndVirial_h = (real *)PMEMemoryFetch(PME_ID_ENERGY_AND_VIRIAL, pme->gpu->energyAndVirialSize, ML_HOST);
        cu_copy_D2H_async(energyAndVirial_h, pme->gpu->energyAndVirial, pme->gpu->energyAndVirialSize, s);
        cudaError_t stat = cudaEventRecord(pme->gpu->syncEnerVirH2D, s);
        CU_RET_ERR(stat, "PME solve energy/virial sync fail");
    }
}

void pme_gpu_get_energy_virial(gmx_pme_t *pme)
{
    cudaStream_t s = pme->gpu->pmeStream;

    struct pme_solve_work_t *work = &pme->solve_work[0];
    real *work_energy_q = &(work->energy_q);
    matrix &work_vir_q = work->vir_q;

    cudaError_t stat = cudaStreamWaitEvent(s, pme->gpu->syncEnerVirH2D, 0);
    CU_RET_ERR(stat, "error while waiting for PME solve");
    real *energyAndVirial_h = (real *)PMEMemoryFetch(PME_ID_ENERGY_AND_VIRIAL, pme->gpu->energyAndVirialSize, ML_HOST);
    real energy = 0.0;
    real virxx = 0.0, virxy = 0.0, virxz = 0.0, viryy = 0.0, viryz = 0.0, virzz = 0.0;

    int j = 0;
    virxx += energyAndVirial_h[j++];
    viryy += energyAndVirial_h[j++];
    virzz += energyAndVirial_h[j++];
    virxy += energyAndVirial_h[j++];
    virxz += energyAndVirial_h[j++];
    viryz += energyAndVirial_h[j++];
    energy += energyAndVirial_h[j++];
    for (j = 0; j < 7; j++)
        GMX_RELEASE_ASSERT(!isnan(energyAndVirial_h[j]), "PME GPU energy calculation is broken");
    //printf("receviing %g\n", energy);


    work_vir_q[XX][XX] = 0.25 * virxx;
    work_vir_q[YY][YY] = 0.25 * viryy;
    work_vir_q[ZZ][ZZ] = 0.25 * virzz;
    work_vir_q[XX][YY] = work_vir_q[YY][XX] = 0.25 * virxy;
    work_vir_q[XX][ZZ] = work_vir_q[ZZ][XX] = 0.25 * virxz;
    work_vir_q[YY][ZZ] = work_vir_q[ZZ][YY] = 0.25 * viryz;

    /* This energy should be corrected for a charged system */
    *work_energy_q = 0.5 * energy;
}

