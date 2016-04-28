#include "gromacs/utility/basedefinitions.h"
#include <cuda.h>
#include "pme-cuda.cuh"
#include "pme-timings.cuh"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/gpu_utils/cudautils.cuh"

pme_gpu_timing::~pme_gpu_timing()
{
#if PME_GPU_TIMINGS
    if (initialized)
    {
        cudaError_t stat;
        stat = cudaEventDestroy(event_start);
        CU_RET_ERR(stat, "PME timing cudaEventDestroy fail");
        stat = cudaEventDestroy(event_stop);
        CU_RET_ERR(stat, "PME timing cudaEventDestroy fail");
        initialized = false;
    }
#endif
}

void pme_gpu_timing::check_init()
{
#if PME_GPU_TIMINGS
    if (!initialized)
    {
        cudaError_t stat;
        stat = cudaEventCreate(&event_start, cudaEventDefault);
        CU_RET_ERR(stat, "PME timing cudaEventCreate fail");
        stat = cudaEventCreate(&event_stop, cudaEventDefault);
        CU_RET_ERR(stat, "PME timing cudaEventCreate fail");
        initialized = true;
    }
#endif
}

void pme_gpu_timing::start_recording(cudaStream_t s)
{
    check_init();
#if PME_GPU_TIMINGS
    cudaError_t stat = cudaEventRecord(event_start, s);
    CU_RET_ERR(stat, "PME timing cudaEventRecord fail");
#endif
}

void pme_gpu_timing::stop_recording(cudaStream_t s)
{
#if PME_GPU_TIMINGS
    cudaError_t stat = cudaEventRecord(event_stop, s);
    CU_RET_ERR(stat, "PME timing cudaEventRecord fail");
    call_count++;
#endif
}

real pme_gpu_timing::get_time_milliseconds()
{
    real milliseconds = 0.0;
#if PME_GPU_TIMINGS
    if (initialized)
    {
        cudaError_t stat = cudaEventElapsedTime(&milliseconds, event_start, event_stop);
        CU_RET_ERR(stat, "PME timing cudaEventElapsedTime fail");
    }
#endif
    return milliseconds;
}

unsigned int pme_gpu_timing::get_call_count()
{
    return call_count;
}

void pme_gpu_timing_start(gmx_pme_t *pme, int ewcsn)
{
    const int i = ewcsn - ewcsPME_INTERPOL_IDX;
    pme->gpu->timingEvents[i].start_recording(pme->gpu->pmeStream);
}

void pme_gpu_timing_stop(gmx_pme_t *pme, int ewcsn)
{
    const int i = ewcsn - ewcsPME_INTERPOL_IDX;
    pme->gpu->timingEvents[i].stop_recording(pme->gpu->pmeStream);
}

void pme_gpu_timing_calculate(gmx_pme_t *pme)
{
    for (int i = 0; i < PME_GPU_STAGES; i++)
    {
        gmx_wallclock_gpu_pme.pme_time[i].t += pme->gpu->timingEvents[i].get_time_milliseconds();
        gmx_wallclock_gpu_pme.pme_time[i].c = pme->gpu->timingEvents[i].get_call_count();
    }
}


void pme_gpu_reset_timings(gmx_pme_t *pme)
{
#if PME_GPU_TIMINGS
    if (pme && pme->bGPU)
    {
        for (int i = 0; i < PME_GPU_STAGES; i++)
        {
            gmx_wallclock_gpu_pme.pme_time[i].t = 0.0;
            gmx_wallclock_gpu_pme.pme_time[i].c = 0;
        }
    }
#endif
}
