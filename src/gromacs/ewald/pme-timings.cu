#include "gromacs/utility/basedefinitions.h"
#include <cuda.h>
#include "pme-cuda.cuh"
#include "pme-timings.cuh"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/gpu_utils/cudautils.cuh"

pme_gpu_timing::pme_gpu_timing()
{
    initialized = false;
    reset();
}

pme_gpu_timing::~pme_gpu_timing()
{
    if (initialized)
    {
        cudaError_t stat;
        stat = cudaEventDestroy(event_start);
        CU_RET_ERR(stat, "PME timing cudaEventDestroy fail");
        stat = cudaEventDestroy(event_stop);
        CU_RET_ERR(stat, "PME timing cudaEventDestroy fail");
        initialized = false;
    }
}

void pme_gpu_timing::enable()
{
    if (!initialized)
    {
        cudaError_t stat;
        stat = cudaEventCreate(&event_start, cudaEventDefault);
        CU_RET_ERR(stat, "PME timing cudaEventCreate fail");
        stat = cudaEventCreate(&event_stop, cudaEventDefault);
        CU_RET_ERR(stat, "PME timing cudaEventCreate fail");
        initialized = true;
    }
}

void pme_gpu_timing::start_recording(cudaStream_t s)
{
    if (initialized)
    {
        cudaError_t stat = cudaEventRecord(event_start, s);
        CU_RET_ERR(stat, "PME timing cudaEventRecord fail");
    }
}

void pme_gpu_timing::stop_recording(cudaStream_t s)
{
    if (initialized)
    {
        cudaError_t stat = cudaEventRecord(event_stop, s);
        CU_RET_ERR(stat, "PME timing cudaEventRecord fail");
        call_count++;
    }
}

void pme_gpu_timing::reset()
{
    total_milliseconds = 0.0;
    call_count = 0;
}

void pme_gpu_timing::update()
{
    if (initialized && (call_count > 0)) // only touched events needed
    {
        real milliseconds = 0.0;
        cudaError_t stat = cudaEventElapsedTime(&milliseconds, event_start, event_stop);
        CU_RET_ERR(stat, "PME timing cudaEventElapsedTime fail");
        total_milliseconds += milliseconds;
    }
}

real pme_gpu_timing::get_total_time_milliseconds()
{
    return total_milliseconds;
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

void pme_gpu_get_timings(gmx_pme_t *pme)
{
    if (pme && pme->bGPU)
    {
        for (int i = 0; i < PME_GPU_STAGES; i++)
        {
            gmx_wallclock_gpu_pme.pme_time[i].t = pme->gpu->timingEvents[i].get_total_time_milliseconds();
            gmx_wallclock_gpu_pme.pme_time[i].c = pme->gpu->timingEvents[i].get_call_count();
        }
    }
}

void pme_gpu_update_timings(gmx_pme_t *pme)
{
    if (pme && pme->bGPU)
    {
        for (int i = 0; i < PME_GPU_STAGES; i++)
            pme->gpu->timingEvents[i].update();
    }
}

void pme_gpu_init_timings(gmx_pme_t *pme)
{
    if (pme && pme->bGPU)
    {
        for (int i = 0; i < PME_GPU_STAGES; i++)
            pme->gpu->timingEvents[i].enable();
    }
}

void pme_gpu_reset_timings(gmx_pme_t *pme)
{
    if (pme && pme->bGPU)
    {
        for (int i = 0; i < PME_GPU_STAGES; i++)
            pme->gpu->timingEvents[i].reset();
    }
}
