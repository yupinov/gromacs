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

#include "th-a.cuh"
#include "check.h"


#include "pme-cuda.h"

#define SQRT_M_PI real(2.0f / M_2_SQRTPI)
//yupinov check if these constants workf

#ifdef DEBUG_PME_TIMINGS_GPU
extern gpu_events gpu_events_solve;
#endif
typedef real *splinevec[DIM];



/* Pascal triangle coefficients used in solve_pme_lj_yzx, only need to do 4 calculations due to symmetry */
static const __constant__ real lb_scale_factor_symm_gpu[] = { 2.0/64, 12.0/64, 30.0/64, 20.0/64 }; //yupinov copied from pme-internal


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

template<
        const gmx_bool bEnerVir,
        const gmx_bool YZXOrdering
        //yupinov - now GPU solve works in a XYZ mode, while original solve worked in YZX order;
        // should be set to true when we do multi-rank GPU PME
        >
__global__ void solve_pme_kernel
(const int localCountMajor, const int localCountMiddle, const int localCountMinor,
 const int localOffsetMinor, const int localOffsetMajor, const int localOffsetMiddle,
 const int localSizeMinor, /*const int localSizeMajor,*/ const int localSizeMiddle,
 const int nMinor, const int nMajor, const int nMiddle,
 const real rxx, const real ryx, const real ryy, const real rzx, const real rzy, const real rzz,
 const real elfac,
 const real * __restrict__ BSplineModuleMinor,
 const real * __restrict__ BSplineModuleMajor,
 const real * __restrict__ BSplineModuleMiddle,
 t_complex * __restrict__ grid,
 const real ewaldcoeff, const real volume,
 real * __restrict__ energy_v, real * __restrict__ virial_v)
{
    const real factor = M_PI * M_PI / (ewaldcoeff * ewaldcoeff);

    int maxkMinor = (nMinor + 1) / 2;
    if (!YZXOrdering) //yupinov - don't really understand it
        maxkMinor = (nMinor + 2) / 2; // [0 25]; -26] fixed, no maxkx required at all
    int maxkMajor = (nMajor + 1) / 2;
    //int maxkz = nz / 2 + 1;

    real energy = 0.0f;
    real virxx = 0.0f, virxy = 0.0f, virxz = 0.0f, viryy = 0.0f, viryz = 0.0f, virzz = 0.0f;

    const int indexMinor = blockIdx.x * blockDim.x + threadIdx.x;
    const int indexMiddle = blockIdx.y * blockDim.y + threadIdx.y;
    const int indexMajor = blockIdx.z * blockDim.z + threadIdx.z;

    if ((indexMajor < localCountMajor) && (indexMiddle < localCountMiddle) && (indexMinor < localCountMinor))
    {
        //yupinov do a reduction!

        //yupinov :localoffset might be a failure point for MPI!

        const int kMajor = indexMajor + localOffsetMajor;
        const real mMajor = (kMajor < maxkMajor) ? kMajor : (kMajor - nMajor);

        const int kMiddle = indexMiddle + localOffsetMiddle;
        real mMiddle = kMiddle;

        if (!YZXOrdering)
            mMiddle = (kMiddle < (nMiddle + 1) / 2) ? kMiddle : (kMiddle - nMiddle);

        const real bMajorMiddle = M_PI * volume * BSplineModuleMajor[kMajor] * BSplineModuleMiddle[kMiddle];

        t_complex *p0 = grid + (indexMajor * localSizeMiddle + indexMiddle) * localSizeMinor + indexMinor;

        /* We should skip the k-space point (0,0,0) */
        /* Note that since here x is the minor index, local_offset[XX]=0 */ //yupinov what

        int kMinor = localOffsetMinor + indexMinor;
        const gmx_bool notZeroPoint = (kMinor > 0 || kMajor > 0 || kMiddle > 0);
        real mMinor, mhxk, mhyk, mhzk, m2k;
        real ets2, struct2, vfactor, ets2vf;

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

        if (notZeroPoint) // this skips just one point in the whole grid!
        {
            if (bEnerVir)
            {
                /* More expensive inner loop, especially because of the storage
                 * of the mh elements in array's.
                 * Because x is the minor grid index, all mh elements
                 * depend on kx for triclinic unit cells.
                 */

                //for (int kx = kxstart; kx < kxend; kx++, p0++)
                {
                    //mx = kMinor < maxkMinor ? kMinor : (kMinor - nMinor);

                    mhxk      = mX * rxx;
                    mhyk      = mX * ryx + mY * ryy;
                    mhzk      = mX * rzx + mY * rzy + mZ * rzz;

                    m2k       = mhxk * mhxk + mhyk * mhyk + mhzk * mhzk;
                    real denom = m2k * bMajorMiddle * BSplineModuleMinor[kMinor];
                    real tmp1  = -factor * m2k;

                    real m2invk = 1.0f / m2k;

                    //calc_exponentials_q_one(elfac, denom, tmp1, eterm);
                    denom = 1.0f / denom;
                    tmp1 = expf(tmp1);
                    real etermk = elfac * tmp1 * denom;

                    real d1 = p0->re;
                    real d2 = p0->im;

                    p0->re  = d1 * etermk;
                    p0->im  = d2 * etermk;

                    struct2 = 2.0f * (d1 * d1 + d2 * d2);

                    real tmp1k = etermk * struct2;

                    ets2     = corner_fac * tmp1k;
                    vfactor  = (factor * m2k + 1.0f) * 2.0f * m2invk;
                    energy  += ets2;

                    ets2vf   = ets2 * vfactor;
                    virxx   = ets2vf * mhxk * mhxk - ets2;
                    virxy   = ets2vf * mhxk * mhyk;
                    virxz   = ets2vf * mhxk * mhzk;
                    viryy   = ets2vf * mhyk * mhyk - ets2;
                    viryz   = ets2vf * mhyk * mhzk;
                    virzz   = ets2vf * mhzk * mhzk - ets2;

                    const int i = (indexMajor * localCountMiddle + indexMiddle) * localCountMinor + indexMinor;
                    energy_v[i] = energy;
                    virial_v[6 * i + 0] = virxx;
                    virial_v[6 * i + 1] = viryy;
                    virial_v[6 * i + 2] = virzz;
                    virial_v[6 * i + 3] = virxy;
                    virial_v[6 * i + 4] = virxz;
                    virial_v[6 * i + 5] = viryz;
                }
            }
            else
            {
                /* We don't need to calculate the energy and the virial.
                 * In this case the triclinic overhead is small.
                 */
                //for (int kx = kxstart; kx < kxend; kx++, p0++)
                //yupinov - now each thread does Re and Im of a grid point => 16 threads in a block? should stride!
                {
                    //mMinor = kMinor < maxkMinor ? kMinor : (kMinor - nMinor);
                    //update mX as well!

                    mhxk      = mX * rxx;
                    mhyk      = mX * ryx + mY * ryy;
                    mhzk      = mX * rzx + mY * rzy + mZ * rzz;
                    m2k       = mhxk * mhxk + mhyk * mhyk + mhzk * mhzk;
                    real denom = m2k * bMajorMiddle * BSplineModuleMinor[kMinor];
                    real tmp1  = -factor * m2k;

                    //calc_exponentials_q_one(elfac, denom, tmp1, eterm);
                    denom = 1.0f / denom;
                    tmp1 = exp(tmp1);
                    real etermk = elfac * tmp1 * denom;

                    real d1 = p0->re;
                    real d2 = p0->im;

                    p0->re = d1 * etermk;
                    p0->im = d2 * etermk;
                }
            }
        }
    }
}

void solve_pme_yzx_gpu(real pme_epsilon_r,
		      int nx, int ny, int nz,
              ivec complex_order, ivec local_ndata, ivec local_offset, ivec local_size,
		      splinevec pme_bsp_mod,
		      matrix work_vir_q, real *work_energy_q,
		      t_complex *grid,
		      real ewaldcoeff, real vol,
		      gmx_bool bEnerVir,
              gmx_pme_t *pme)
{
     /* do recip sum over local cells in grid */
    const gmx_bool YZXOrdering = false; //yupinov fix pecularities in solve
    /* true: y major, z middle, x minor or continuous - the CPU FFT way */
    /* false: x major, y middle, z minor - the single rank GPU cuFFT way */
    const int minorDim = !YZXOrdering ? ZZ : XX;
    const int middleDim = !YZXOrdering ? YY : ZZ;
    const int majorDim = !YZXOrdering ? XX : YY;
    const int nMinor = !YZXOrdering ? pme->nkz : pme->nkx;
    const int nMajor = !YZXOrdering ? pme->nkx : pme->nky;
    const int nMiddle = !YZXOrdering ? pme->nky : pme->nkz;


    const int thread = 0;
    cudaStream_t s = pme->gpu->pmeStream;

    real energy = 0.0;
    real virxx = 0.0, virxy = 0.0, virxz = 0.0, viryy = 0.0, viryz = 0.0, virzz = 0.0;
    const real elfac = ONE_4PI_EPS0 / pme_epsilon_r;

    real rxx = pme->recipbox[XX][XX];
    real ryx = pme->recipbox[YY][XX];
    real ryy = pme->recipbox[YY][YY];
    real rzx = pme->recipbox[ZZ][XX];
    real rzy = pme->recipbox[ZZ][YY];
    real rzz = pme->recipbox[ZZ][ZZ];

    /* Dimensions should be identical for A/B grid, so we just use A here */

    /*
      TODO: Dimensions are passed in for now. call complex limits elsewhere?
    gmx_parallel_3dfft_complex_limits(pme->pfft_setup[PME_GRID_QA],
                                      complex_order,
                                      local_ndata,
                                      local_offset,
                                      local_size);
    gmx_parallel_3dfft_complex_limits_gpu(pme->pfft_setup_gpu[PME_GRID_QA],
                                      complex_order,
                                      local_ndata,
                                      local_offset,
                                      local_size);
    */

    const int blockSize = 4 * warp_size;
    // GTX 660 Ti, 20160310
    // number, occupancy, time from cudaEvents:
    // 0.5  0.24  0.181
    // 1    0.24  0.110
    // 2    0.43  0.082
    // 4    0.89  0.080
    // 8    0.80  0.084

    // Z-dimension is too small in CUDA limitations (64 on CC30?), so instead of major-middle-minor sizing we do minor-middle-major
    dim3 blocks((local_ndata[minorDim] + blockSize - 1) / blockSize, local_ndata[middleDim], local_ndata[majorDim]);
    dim3 threads(blockSize, 1, 1);

    //yupinov align minor dimension with cachelines!

    const int n = local_ndata[majorDim] * local_ndata[middleDim] * local_ndata[minorDim];
    const int grid_n = local_size[majorDim] * local_size[middleDim] * local_size[minorDim];
    const int grid_size = grid_n * sizeof(t_complex);

    real *bspModMinor_d = PMEFetchAndCopyRealArray(PME_ID_BSP_MOD_MINOR, thread, pme_bsp_mod[minorDim], nMinor * sizeof(real), ML_DEVICE, s);
    real *bspModMajor_d = PMEFetchAndCopyRealArray(PME_ID_BSP_MOD_MAJOR, thread, pme_bsp_mod[majorDim], nMajor * sizeof(real), ML_DEVICE, s);
    real *bspModMiddle_d = PMEFetchAndCopyRealArray(PME_ID_BSP_MOD_MIDDLE, thread, pme_bsp_mod[middleDim], nMiddle * sizeof(real), ML_DEVICE, s);

    const int energySize = n * sizeof(real);
    const int virialSize = 6 * n * sizeof(real);
    real *energy_d = PMEFetchRealArray(PME_ID_ENERGY, thread, energySize, ML_DEVICE);
    real *virial_d = PMEFetchRealArray(PME_ID_VIRIAL, thread, virialSize, ML_DEVICE);

    t_complex *grid_d = PMEFetchComplexArray(PME_ID_COMPLEX_GRID, thread, grid_size, ML_DEVICE);
    if (!pme->gpu->keepGPUDataBetweenR2CAndSolve)
        PMECopy(grid_d, grid, grid_size, ML_DEVICE, s);

#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_start(gpu_events_solve, s);
#endif
    if (bEnerVir)
        solve_pme_kernel<TRUE, YZXOrdering> <<<blocks, threads, 0, s>>>
          (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim],
           local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
           local_size[minorDim], /*local_size[majorDim],*/ local_size[middleDim],
           nMinor, nMajor, nMiddle, rxx, ryx, ryy, rzx, rzy, rzz,
           elfac,
           bspModMinor_d, bspModMajor_d, bspModMiddle_d,
           grid_d, ewaldcoeff, vol,
           energy_d, virial_d);
    else
        solve_pme_kernel<FALSE, YZXOrdering> <<<blocks, threads, 0, s>>>
          (local_ndata[majorDim], local_ndata[middleDim], local_ndata[minorDim ],
           local_offset[minorDim], local_offset[majorDim], local_offset[middleDim],
           local_size[minorDim], /*local_size[majorDim],*/local_size[middleDim],
           nMinor, nMajor, nMiddle, rxx, ryx, ryy, rzx, rzy, rzz,
           elfac,
           bspModMinor_d, bspModMajor_d, bspModMiddle_d,
           grid_d, ewaldcoeff, vol,
           energy_d, virial_d);
    CU_LAUNCH_ERR("solve_pme_kernel");
#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_stop(gpu_events_solve, s, ewcsPME_SOLVE, 0);
#endif

    if (!pme->gpu->keepGPUDataBetweenSolveAndC2R)
    {
        PMECopy(grid, grid_d, grid_size, ML_HOST, s);
    }

    if (bEnerVir)
    {
        real *energy_h = PMEFetchAndCopyRealArray(PME_ID_ENERGY, thread, energy_d, energySize, ML_HOST, s);
        real *virial_h = PMEFetchAndCopyRealArray(PME_ID_VIRIAL, thread, virial_d, virialSize, ML_HOST, s);
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

        work_vir_q[XX][XX] = 0.25 * virxx;
        work_vir_q[YY][YY] = 0.25 * viryy;
        work_vir_q[ZZ][ZZ] = 0.25 * virzz;
        work_vir_q[XX][YY] = work_vir_q[YY][XX] = 0.25 * virxy;
        work_vir_q[XX][ZZ] = work_vir_q[ZZ][XX] = 0.25 * virxz;
        work_vir_q[YY][ZZ] = work_vir_q[ZZ][YY] = 0.25 * viryz;

        /* This energy should be corrected for a charged system */
        *work_energy_q = 0.5 * energy;
    }
    /* Return the loop count */
    //return local_ndata[YY]*local_ndata[XX]; //yupinov why
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
 t_complex *grid_v, gmx_bool bLB,
 real ewaldcoeff, real vol,
 gmx_bool bEnerVir,
 real *energy_v, real *virial_v);


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

    //cudaError_t stat = cudaMemcpyToSymbol( sqrt_M_PI_d, &sqrt_M_PI, sizeof(real));
    //CU_RET_ERR(stat, "solve cudaMemcpyToSymbol");

    const int block_size = warp_size;
    int n = iyz1 - iyz0;
    int n_blocks = (n + block_size - 1) / block_size;

#define MAGIC_GRID_NUMBER 6
    //yupinov

    int grid_n = local_size[YY] * local_size[ZZ] * local_size[XX];
    int grid_size = grid_n * sizeof(t_complex);
    t_complex *grid_d = PMEFetchComplexArray(PME_ID_COMPLEX_GRID, thread, grid_size * MAGIC_GRID_NUMBER, ML_DEVICE); //6 grids!
    real *pme_bsp_mod_x_d = PMEFetchAndCopyRealArray(PME_ID_BSP_MOD_MINOR, thread, pme_bsp_mod[XX], nx * sizeof(real), ML_DEVICE, s);
    real *pme_bsp_mod_y_d = PMEFetchAndCopyRealArray(PME_ID_BSP_MOD_MAJOR, thread, pme_bsp_mod[YY], ny * sizeof(real), ML_DEVICE, s);
    real *pme_bsp_mod_z_d = PMEFetchAndCopyRealArray(PME_ID_BSP_MOD_MIDDLE, thread, pme_bsp_mod[ZZ], nz * sizeof(real), ML_DEVICE, s);
    int energy_size = n * sizeof(real);
    int virial_size = 6 * n * sizeof(real);
    real *energy_d = PMEFetchRealArray(PME_ID_ENERGY, thread, energy_size, ML_DEVICE);
    real *virial_d = PMEFetchRealArray(PME_ID_VIRIAL, thread, virial_size, ML_DEVICE);
    for (int ig = 0; ig < MAGIC_GRID_NUMBER; ++ig)
        PMECopy(grid_d + ig * grid_n, grid[ig], grid_size, ML_DEVICE, s);
#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_start(gpu_events_solve, s);
#endif
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
#ifdef DEBUG_PME_TIMINGS_GPU
    events_record_stop(gpu_events_solve, s, ewcsPME_SOLVE, 0);
#endif
    for (int ig = 0; ig < MAGIC_GRID_NUMBER; ++ig)
        PMECopy(grid[ig], grid_d + ig * grid_n, grid_size, ML_HOST, s);

    if (bEnerVir) //yupinov check if it works!
    {
        real *energy_h = PMEFetchAndCopyRealArray(PME_ID_ENERGY, thread, energy_d, energy_size, ML_HOST, s);
        real *virial_h = PMEFetchAndCopyRealArray(PME_ID_VIRIAL, thread, virial_d, virial_size, ML_HOST, s);
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


__global__ void solve_pme_lj_yzx_iyz_loop_kernel
(int iyz0, int iyz1, int local_ndata_ZZ, int local_ndata_XX,
 int local_offset_XX, int local_offset_YY, int local_offset_ZZ,
 int local_size_XX, int local_size_YY, int local_size_ZZ,
 int nx, int ny, int nz,
 real rxx, real ryx, real ryy, real rzx, real rzy, real rzz,
 //real ,
 //splinevec pme_bsp_mod,
 real *pme_bsp_mod_XX, real *pme_bsp_mod_YY, real *pme_bsp_mod_ZZ,
 t_complex *grid_v, gmx_bool bLB,
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
            t_complex *p0 = grid_v/*[0]*/ + iy*local_size_ZZ*local_size_XX + iz*local_size_XX;
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
                    real d1      = p0->re;
                    real d2      = p0->im;

                    eterm   = tmp1k;
                    vterm   = tmp2k;
                    p0->re  = d1*eterm;
                    p0->im  = d2*eterm;

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

                        t_complex *p0k    = grid_v/*[ig]*/ + ig*grid_size + iy*local_size_ZZ*local_size_XX + iz*local_size_XX + (kx - kxstart);
                        t_complex *p1k    = grid_v/*[6-ig]*/ + (6-ig)*grid_size + iy*local_size_ZZ*local_size_XX + iz*local_size_XX + (kx - kxstart);
                        scale = 2.0f * lb_scale_factor_symm_gpu[ig];
                        struct2k += scale*(p0k->re*p1k->re + p0k->im*p1k->im);
                    }
                    for (int ig = 0; ig <= 6; ++ig)
                    {
                        //t_complex *p0;

                        t_complex *p0k = grid_v/*[ig]*/ + ig*grid_size + iy*local_size_ZZ*local_size_XX + iz*local_size_XX + (kx - kxstart);

                        real d1     = p0k->re;
                        real d2     = p0k->im;

                        eterm  = tmp1k;
                        p0k->re = d1*eterm;
                        p0k->im = d2*eterm;
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

                    t_complex *p0k = grid_v/*[ig]*/ + ig*grid_size + iy*local_size_ZZ*local_size_XX + iz*local_size_XX + (kx - kxstart);

                    real d1      = p0k->re;
                    real d2      = p0k->im;

                    eterm   = tmp1k;

                    p0k->re  = d1*eterm;
                    p0k->im  = d2*eterm;
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
