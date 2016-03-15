#include "pme.h"
#include "pme-internal.h"

#include "gromacs/fft/fft.h"
#include "gromacs/fft/parallel_3dfft.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/real.h"
#include "gromacs/math/vectypes.h"

#include "check.h"

#include <cuda.h>
#include <cufft.h>

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"

#ifdef DEBUG_PME_TIMINGS_GPU
extern gpu_events gpu_events_fft_r2c;
extern gpu_events gpu_events_fft_c2r;
#endif

#include "pme-cuda.h"

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

    setup->size_real[XX] = setup->ndata_real[XX];
    setup->size_real[YY] = setup->ndata_real[YY];
    setup->size_real[ZZ] = (setup->ndata_real[ZZ] / 2 + 1) * 2;

    setup->size_complex[XX] = setup->ndata_real[XX];
    setup->size_complex[YY] = setup->ndata_real[YY];
    setup->size_complex[ZZ] = setup->ndata_real[ZZ] / 2 + 1;

    const int gridSizeComplex = setup->size_complex[XX] * setup->size_complex[YY] * setup->size_complex[ZZ];
    const int gridSizeReal = setup->size_real[XX] * setup->size_real[YY] * setup->size_real[ZZ];

    setup->rdata = PMEFetchRealArray(PME_ID_REAL_GRID, 0, gridSizeReal * sizeof(cufftReal), ML_DEVICE);
    setup->cdata = (cufftComplex *)PMEFetchComplexArray(PME_ID_COMPLEX_GRID, 0, gridSizeComplex * sizeof(cufftComplex), ML_DEVICE);

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
    local_ndata[0] = setup->ndata_real[0];
    local_ndata[1] = setup->ndata_real[1];
    local_ndata[2] = setup->ndata_real[2];
    local_size[0] = setup->size_real[0];
    local_size[1] = setup->size_real[1];
    local_size[2] = setup->size_real[2];

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
    local_ndata[0] = setup->ndata_real[0];
    local_ndata[1] = setup->ndata_real[1];
    local_ndata[2] = setup->ndata_real[2] / 2 + 1;
    local_size[0] = setup->size_complex[0];
    local_size[1] = setup->size_complex[1];
    local_size[2] = setup->size_complex[2];
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

    int x = setup->ndata_real[0], y = setup->ndata_real[1], z = setup->ndata_real[2];

    if (dir == GMX_FFT_REAL_TO_COMPLEX)
    {      
        if (!pme->gpu->keepGPUDataBetweenSpreadAndR2C)
            PMECopy(setup->rdata, setup->real_data, x * y * (z / 2 + 1) * 2 * sizeof(real), ML_DEVICE, s);
        /*
        //yupinov hack for padded data
        {
            cufftReal *dest = setup->rdata;
            real *src = setup->real_data;
            for (int xi = 0; xi < x; xi++)
                for (int yi = 0; yi < y; yi++)
                 {
                    int size = z;
                    int stripe = (z / 2 + 1) * 2;
                    stat = cudaMemcpy(dest, src, size * sizeof(real), cudaMemcpyHostToDevice);
                    CU_RET_ERR(stat, "cudaMemcpy R2C error");
                    dest += size;
                    src += stripe;
                 }
        }
        */


        #ifdef DEBUG_PME_TIMINGS_GPU
        events_record_start(gpu_events_fft_r2c, s);
        #endif
        cufftResult_t result = cufftExecR2C(setup->planR2C, setup->rdata, setup->cdata);
        if (result)
            fprintf(stderr, "cufft R2C error %d\n", result);
        #ifdef DEBUG_PME_TIMINGS_GPU
        events_record_stop(gpu_events_fft_r2c, s, ewcsPME_FFT_R2C, 0);
        #endif
    }
    else
    {
        if (!pme->gpu->keepGPUDataBetweenSolveAndC2R)
            PMECopy(setup->cdata, setup->complex_data, x * y * (z / 2 + 1) * sizeof(t_complex), ML_DEVICE, s);
        #ifdef DEBUG_PME_TIMINGS_GPU
        events_record_start(gpu_events_fft_c2r, s);
        #endif
        cufftResult_t result = cufftExecC2R(setup->planC2R, setup->cdata, setup->rdata);
        if (result)
            fprintf(stderr, "cufft C2R error %d\n", result);
        #ifdef DEBUG_PME_TIMINGS_GPU
        events_record_stop(gpu_events_fft_c2r, s, ewcsPME_FFT_C2R, 0);
        #endif
    }

    if (dir == GMX_FFT_REAL_TO_COMPLEX)
    {
        if (!pme->gpu->keepGPUDataBetweenR2CAndSolve)
            PMECopy(setup->complex_data, setup->cdata, x * y * (z / 2 + 1) * sizeof(t_complex), ML_HOST, s);
    }
    else
    {
        if (!pme->gpu->keepGPUDataBetweenC2RAndGather)
            PMECopy(setup->real_data, setup->rdata, x * y * (z / 2 + 1) * 2 * sizeof(real), ML_HOST, s);
        /*
        //yupinov hack for padded data
        {
            real *dest = setup->real_data;
            cufftReal *src = setup->rdata;
            for (int xi = 0; xi < x; xi++)
                for (int yi = 0; yi < y; yi++)
                 {
                    int size = z;
                    int stripe = (z / 2 + 1) * 2;
                    stat = cudaMemcpy(dest, src, size * sizeof(real), cudaMemcpyDeviceToHost);
                    CU_RET_ERR(stat, "cudaMemcpy C2R error");
                    dest += stripe;
                    src += size;
                 }
        }
        */
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
