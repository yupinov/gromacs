#include "gromacs/utility/basedefinitions.h"
#include <cuda.h>
#include "pme-cuda.cuh"
#include "pme-timings.cuh"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/gpu_utils/cudautils.cuh"

void pme_gpu_timing_start(gmx_pme_t *pme, int ewcsn)
{
#if PME_GPU_TIMINGS
    const int i = ewcsn - ewcsPME_INTERPOL_IDX;
    pme_gpu_timing *event = &pme->gpu->timingEvents[i];
    cudaError_t stat;
    if (!event->created)
    {
        stat = cudaEventCreate(&event->event_start);
        CU_RET_ERR(stat, "?");
        stat = cudaEventCreate(&event->event_stop);
        CU_RET_ERR(stat, "?");
        event->created = true;
    }
    stat = cudaEventRecord(event->event_start, pme->gpu->pmeStream);
    CU_RET_ERR(stat, "?");
#endif
}

void pme_gpu_timing_stop(gmx_pme_t *pme, int ewcsn)
{
#if PME_GPU_TIMINGS
    const int i = ewcsn - ewcsPME_INTERPOL_IDX;
    pme_gpu_timing *event = &pme->gpu->timingEvents[i];
    cudaError_t stat;
    stat = cudaEventRecord(event->event_stop, pme->gpu->pmeStream);
    CU_RET_ERR(stat, "?");
    ++gmx_wallclock_gpu_pme.pme_time[i].c;
#endif
}

void pme_gpu_timing_calculate(gmx_pme_t *pme)
{
#if PME_GPU_TIMINGS
    cudaError_t stat;
    for (int i = 0; i < PME_GPU_STAGES; i++)
    {
        if (pme->gpu->timingEvents[i].created)
        {
            real milliseconds = 0;
            stat = cudaEventElapsedTime(&milliseconds, pme->gpu->timingEvents[i].event_start, pme->gpu->timingEvents[i].event_stop);
            CU_RET_ERR(stat, "?");
            gmx_wallclock_gpu_pme.pme_time[i].t += milliseconds;
        }
    }
#endif
}
