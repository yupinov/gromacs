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

#include "th-a.cuh"


struct gmx_parallel_3dfft_gpu
{
    real *real_data;
    t_complex *complex_data;

    /* unused */
    MPI_Comm                  comm[2];
    gmx_bool                  bReproducible;
    int                       nthreads;

    ivec                      complex_order;
    ivec                      local_ndata;
    ivec                      local_offset;
    ivec                      local_size;

    int n[3];
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
    setup->n[0] = ndata[0];
    setup->n[1] = ndata[1];
    setup->n[2] = ndata[2]; //yupinov ZZ

    int x = setup->n[0], y = setup->n[1], z = setup->n[2];

    setup->rdata = th_a(TH_ID_REAL_GRID, 0, x * y * (z / 2 + 1) * 2 * sizeof(cufftReal), TH_LOC_CUDA);
    setup->cdata = (cufftComplex *)th_c(TH_ID_COMPLEX_GRID, 0, x * y * (z / 2 + 1) * 2 * sizeof(cufftReal), TH_LOC_CUDA);

    *pfft_setup = setup;

    cufftResult_t result;
    /*
    result = cufftPlan3d(&setup->planR2C, setup->n[0], setup->n[1], setup->n[2], CUFFT_R2C);
    if (result != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cufft planR2C error %d\n", result);
        setup = NULL; //yupinov FIX
    }

    result = cufftPlan3d(&setup->planC2R, setup->n[0], setup->n[1], setup->n[2], CUFFT_C2R);
    if (result != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cufft planC2R error %d\n", result);
        setup = NULL; // FIX
    }
    */

    int rembed[3];
    rembed[0] = setup->n[XX];
    rembed[1] = setup->n[YY];
    rembed[2] = setup->n[ZZ];
    rembed[2] = (rembed[2] / 2 + 1) * 2;
    int cembed[3];
    cembed[0] = setup->n[XX];
    cembed[1] = setup->n[YY];
    cembed[2] = setup->n[ZZ];
    cembed[2] = (cembed[2] / 2 + 1);

    int rank = 3, batch = 1;

    result = cufftPlanMany(&setup->planR2C, rank, setup->n,
                                       rembed, 1, rembed[0] * rembed[1] * rembed[2],
                                       cembed, 1, cembed[0] * cembed[1] * cembed[2],
                                       CUFFT_R2C,
                                      batch);
    if (result != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cufft planR2RC error %d\n", result); //yupinov throw fatalerror
        setup = NULL; // FIX
    }

    result = cufftPlanMany(&setup->planC2R, rank, setup->n,
                                       cembed, 1, cembed[0] * cembed[1] * cembed[2],
                                       rembed, 1, rembed[0] * rembed[1] * rembed[2],
                                       CUFFT_C2R,
                                       batch);
    if (result != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cufft planC2R error %d\n", result);
        setup = NULL; // FIX
    }

    cudaStream_t s = pme->gpu->pmeStream;
    result = cufftSetStream(setup->planR2C, s);
    if (result != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cufft planR2C error %d\n", result);
        setup = NULL;
    }
    result = cufftSetStream(setup->planC2R, s);
    if (result != CUFFT_SUCCESS)
    {
        fprintf(stderr, "cufft planC2R stream error %d\n", result);
        setup = NULL;
    }
}

void gmx_parallel_3dfft_real_limits_gpu(gmx_parallel_3dfft_gpu_t      pfft_setup,
                                       ivec                      local_ndata,
                                       ivec                      local_offset,
                                       ivec                      local_size)
{
    //fprintf(stderr, "3dfft_real_limits_gpu\n");
    gmx_parallel_3dfft_gpu_t setup = pfft_setup;
    setup->local_ndata[0] = local_ndata[0];
    setup->local_ndata[1] = local_ndata[1];
    setup->local_ndata[2] = local_ndata[2];
    setup->local_offset[0] = local_offset[0];
    setup->local_offset[1] = local_offset[1];
    setup->local_offset[2] = local_offset[2];
    setup->local_size[0] = local_size[0];
    setup->local_size[1] = local_size[1];
    setup->local_size[2] = local_size[2];
}

void gmx_parallel_3dfft_complex_limits_gpu(gmx_parallel_3dfft_gpu_t      pfft_setup,
                                          ivec                      complex_order,
                                          ivec                      local_ndata,
                                          ivec                      local_offset,
                                          ivec                      local_size)
{
    //yupinov why are they here
    //fprintf(stderr, "3dfft_complex_limits_gpu\n");
    gmx_parallel_3dfft_gpu_t setup = pfft_setup;
    setup->complex_order[0] = complex_order[0];
    setup->complex_order[1] = complex_order[1];
    setup->complex_order[2] = complex_order[2];
    setup->local_ndata[0] = local_ndata[0];
    setup->local_ndata[1] = local_ndata[1];
    setup->local_ndata[2] = local_ndata[2];
    setup->local_offset[0] = local_offset[0];
    setup->local_offset[1] = local_offset[1];
    setup->local_offset[2] = local_offset[2];
    setup->local_size[0] = local_size[0];
    setup->local_size[1] = local_size[1];
    setup->local_size[2] = local_size[2];
}

void gmx_parallel_3dfft_execute_gpu(gmx_parallel_3dfft_gpu_t    pfft_setup,
                                   enum gmx_fft_direction  dir,
                                   int                     thread,
                                   gmx_wallcycle_t         wcycle,
                                   gmx_pme_t *pme,
                                   t_complex **complexFFTGridSavedOnDevice)
{
    cudaStream_t s = pme->gpu->pmeStream;

    gmx_parallel_3dfft_gpu_t setup = pfft_setup;

    int x = setup->n[0], y = setup->n[1], z = setup->n[2];

    if (dir == GMX_FFT_REAL_TO_COMPLEX)
    {      
        th_cpy(setup->rdata, setup->real_data, x * y * (z / 2 + 1) * 2 * sizeof(real), TH_LOC_CUDA, s);
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
        //yupinov no second transfer
        th_cpy(setup->cdata, setup->complex_data, x * y * (z / 2 + 1) * sizeof(t_complex), TH_LOC_CUDA, s);
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
        cufftComplex *complexFFTGrid = setup->cdata;
        if (!complexFFTGridSavedOnDevice)
            th_cpy(setup->complex_data, complexFFTGrid, x * y * (z / 2 + 1) * sizeof(t_complex), TH_LOC_HOST, s);
        else
            *complexFFTGridSavedOnDevice = (t_complex *)complexFFTGrid;
    }
    else
    {
        th_cpy(setup->real_data, setup->rdata, x * y * (z / 2 + 1) * 2 * sizeof(real), TH_LOC_HOST, s);
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
    // FIX destroy plans after
    //cufftDestroy(setup->plan);
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
