#include <cufft.h>
#include "check.h"
#include "pme-cuda.h"

#include "gromacs/utility/gmxassert.h"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"

#ifdef DEBUG_PME_TIMINGS_GPU
extern gpu_events gpu_events_fft_r2c;
extern gpu_events gpu_events_fft_c2r;
#endif



struct gmx_parallel_3dfft_gpu
{
    real *real_data;
    t_complex *complex_data;

    /* unused */
    MPI_Comm                  comm[2];
    gmx_bool                  bReproducible;
    int                       nthreads;

    ivec                      complex_order;
    ivec                      local_offset;

    ivec ndata_real;
    ivec size_real;
    ivec size_complex;

    cufftHandle planR2C;
    cufftHandle planC2R;
    cufftReal *rdata;
    cufftComplex *cdata;
};

//yupinov warn against double precision

void gmx_parallel_3dfft_init_gpu(gmx_parallel_3dfft_gpu_t *pfft_setup,
                                   ivec                      ndata,
                                   real **real_data,
                                   t_complex **complex_data,
                                   MPI_Comm                  comm[2],
gmx_bool                  bReproducible,
int                       nthreads,
gmx_pme_t *pme)
{
    cufftResult_t result;
    gmx_parallel_3dfft_gpu_t setup = new gmx_parallel_3dfft_gpu();

    //yupinov FIXME: this copies the already setup pointer, to check them after execute

    setup->real_data = *real_data;

    setup->complex_data = *complex_data;

    setup->comm[0] = comm[0];
    setup->comm[1] = comm[1];
    setup->bReproducible = bReproducible;
    setup->nthreads = nthreads;

    /*
    // (local pme and fft differs only by overlap (and pme > fft))
    pmeidx = ix*(local_pme[YY]*local_pme[ZZ])+iy*(local_pme[ZZ])+iz;
    fftidx = ix*(local_fft[YY]*local_fft[ZZ])+iy*(local_fft[ZZ])+iz;
    fftgrid[fftidx] = pmegrid[pmeidx];
    // TODO: align cufft minor dim to 128 bytes
   */
    setup->ndata_real[0] = ndata[XX];
    setup->ndata_real[1] = ndata[YY];
    setup->ndata_real[2] = ndata[ZZ]; //yupinov ZZ

    *pfft_setup = setup;

    /*
    ndata[XX] += pme->pme_order - 1;
    ndata[YY] += pme->pme_order - 1;
    ndata[ZZ] += pme->pme_order - 1;
    */
    if (pme->bGPUSingle)
    {
        ndata[XX] = pme->pmegrid_nx;
        ndata[YY] = pme->pmegrid_ny;
        ndata[ZZ] = pme->pmegrid_nz;
    }
    else
        gmx_fatal(FARGS, "FFT size choice not implemented");

    /*
    setup->size_real[XX] = ndata[XX];
    setup->size_real[YY] = ndata[YY];
    setup->size_real[ZZ] = (ndata[ZZ] / 2 + 1) * 2;
    const int alignment = 1; //warp_size; //yupinov change it so it's in X for YZX solve (what?)
    setup->size_real[ZZ] = (setup->size_real[ZZ] + alignment - 1) / alignment * alignment;
    */

    memcpy(setup->size_real, ndata, sizeof(setup->size_real));

    memcpy(setup->size_complex, setup->size_real, sizeof(setup->size_real));
    GMX_ASSERT(setup->size_complex[ZZ] % 2 == 0, "odd inplace cuFFT dimension size");
    setup->size_complex[ZZ] /= 2;

    const int gridSizeComplex = setup->size_complex[XX] * setup->size_complex[YY] * setup->size_complex[ZZ];
    const int gridSizeReal = setup->size_real[XX] * setup->size_real[YY] * setup->size_real[ZZ];

    setup->rdata = PMEFetchRealArray(PME_ID_REAL_GRID, 0, gridSizeReal * sizeof(cufftReal), ML_DEVICE);
    setup->cdata = (cufftComplex *)PMEFetchComplexArray(PME_ID_COMPLEX_GRID, 0, gridSizeComplex * sizeof(cufftComplex), ML_DEVICE);

    //yupinov hack
    /*
    //we want CPU FFT grids to be of same size, to include the overlap
    {
        free(setup->real_data);
        *real_data = setup->real_data = PMEFetchRealArray(PME_ID_REAL_GRID, 0, gridSizeReal * sizeof(cufftReal), ML_HOST);
        free(setup->complex_data);
        *complex_data = setup->complex_data = PMEFetchComplexArray(PME_ID_COMPLEX_GRID, 0, gridSizeComplex * sizeof(cufftComplex), ML_HOST);
    }
    */
    const int rank = 3, batch = 1;

    /*
    result = cufftPlan3d(&setup->planR2C, setup->ndata_real[XX], setup->ndata_real[YY], setup->ndata_real[ZZ], CUFFT_R2C);
    if (result != CUFFT_SUCCESS)
        gmx_fatal(FARGS, "cufftPlan3d R2C error %d\n", result);

    result = cufftPlan3d(&setup->planC2R, setup->ndata_real[XX], setup->ndata_real[YY], setup->ndata_real[ZZ], CUFFT_C2R);
    if (result != CUFFT_SUCCESS)
        gmx_fatal(FARGS, "cufftPlan3d C2R error %d\n", result);
    */

    result = cufftPlanMany(&setup->planR2C, rank, setup->ndata_real,
                                       setup->size_real, 1, gridSizeReal,
                                       setup->size_complex, 1, gridSizeComplex,
                                       CUFFT_R2C,
                                       batch);
    if (result != CUFFT_SUCCESS)
        gmx_fatal(FARGS, "cufftPlanMany R2C error %d\n", result);

    result = cufftPlanMany(&setup->planC2R, rank, setup->ndata_real,
                                       setup->size_complex, 1, gridSizeComplex,
                                       setup->size_real, 1, gridSizeReal,
                                       CUFFT_C2R,
                                       batch);
    if (result != CUFFT_SUCCESS)
        gmx_fatal(FARGS, "cufftPlanMany C2R error %d\n", result);

    cudaStream_t s = pme->gpu->pmeStream;
    result = cufftSetStream(setup->planR2C, s);
    if (result != CUFFT_SUCCESS)
        gmx_fatal(FARGS, "cufftSetStream R2C error %d\n", result);

    result = cufftSetStream(setup->planC2R, s);
    if (result != CUFFT_SUCCESS)
        gmx_fatal(FARGS, "cufftSetStream C2R error %d\n", result);
}

void gmx_parallel_3dfft_real_limits_gpu(gmx_parallel_3dfft_gpu_t      setup,
                                       ivec                      local_ndata,
                                       ivec                      local_offset,
                                       ivec                      local_size)
{
    if (local_ndata)
        memcpy(local_ndata, setup->ndata_real, sizeof(setup->ndata_real));
    if (local_size)
        memcpy(local_size, setup->size_real, sizeof(setup->size_real));

    //yupinov
    setup->local_offset[0] = local_offset[0];
    setup->local_offset[1] = local_offset[1];
    setup->local_offset[2] = local_offset[2];
}

void gmx_parallel_3dfft_complex_limits_gpu(gmx_parallel_3dfft_gpu_t      setup,
                                          ivec                      complex_order,
                                          ivec                      local_ndata,
                                          ivec                      local_offset,
                                          ivec                      local_size)
{
    if (local_ndata)
    {
        memcpy(local_ndata, setup->ndata_real, sizeof(setup->ndata_real));
        local_ndata[ZZ] = local_ndata[ZZ] / 2 + 1;
    }
    if (local_size)
        memcpy(local_size, setup->size_complex, sizeof(setup->size_complex));

    //yupinov why are they here
    setup->complex_order[0] = complex_order[0];
    setup->complex_order[1] = complex_order[1];
    setup->complex_order[2] = complex_order[2];
    setup->local_offset[0] = local_offset[0];
    setup->local_offset[1] = local_offset[1];
    setup->local_offset[2] = local_offset[2];
}

void gmx_parallel_3dfft_execute_gpu(gmx_parallel_3dfft_gpu_t    pfft_setup,
                                   enum gmx_fft_direction  dir,
                                   gmx_pme_t *pme)
{
    cudaStream_t s = pme->gpu->pmeStream;

    gmx_parallel_3dfft_gpu_t setup = pfft_setup;

    const int gridSizeComplex = setup->size_complex[XX] * setup->size_complex[YY] * setup->size_complex[ZZ] * sizeof(cufftComplex);
    const int gridSizeReal = setup->size_real[XX] * setup->size_real[YY] * setup->size_real[ZZ] * sizeof(cufftReal);

    if (dir == GMX_FFT_REAL_TO_COMPLEX)
    {      
        if (!pme->gpu->keepGPUDataBetweenSpreadAndR2C)
            PMECopy(setup->rdata, setup->real_data, gridSizeReal, ML_DEVICE, s);

        #ifdef DEBUG_PME_TIMINGS_GPU
        events_record_start(gpu_events_fft_r2c, s);
        #endif
        cufftResult_t result = cufftExecR2C(setup->planR2C, setup->rdata, setup->cdata);
        #ifdef DEBUG_PME_TIMINGS_GPU
        events_record_stop(gpu_events_fft_r2c, s, ewcsPME_FFT_R2C, 0);
        #endif
        if (result)
            fprintf(stderr, "cufft R2C error %d\n", result);
    }
    else
    {
        if (!pme->gpu->keepGPUDataBetweenSolveAndC2R)
            PMECopy(setup->cdata, setup->complex_data, gridSizeComplex, ML_DEVICE, s);
        #ifdef DEBUG_PME_TIMINGS_GPU
        events_record_start(gpu_events_fft_c2r, s);
        #endif
        cufftResult_t result = cufftExecC2R(setup->planC2R, setup->cdata, setup->rdata);
        #ifdef DEBUG_PME_TIMINGS_GPU
        events_record_stop(gpu_events_fft_c2r, s, ewcsPME_FFT_C2R, 0);
        #endif
        if (result)
            fprintf(stderr, "cufft C2R error %d\n", result);
    }

    if (dir == GMX_FFT_REAL_TO_COMPLEX)
    {
        if (!pme->gpu->keepGPUDataBetweenR2CAndSolve)
            PMECopy(setup->complex_data, setup->cdata, gridSizeComplex, ML_HOST, s);
    }
    else
    {
        if (!pme->gpu->keepGPUDataBetweenC2RAndGather)
            PMECopy(setup->real_data, setup->rdata, gridSizeReal, ML_HOST, s);
    }
}

void gmx_parallel_3dfft_destroy_gpu(gmx_parallel_3dfft_gpu_t pfft_setup)
{
    //fprintf(stderr, "3dfft_destroy_gpu\n");
    gmx_parallel_3dfft_gpu_t setup = pfft_setup;

    cufftDestroy(setup->planR2C);
    cufftDestroy(setup->planC2R);
    printf("free\n"); //yupinov
    cudaError_t stat = cudaFree((void **)setup->rdata);
    CU_RET_ERR(stat, "cudaFree error");
    stat = cudaFree((void **)setup->cdata);
    CU_RET_ERR(stat, "cudaFree error");

    delete setup;
}
