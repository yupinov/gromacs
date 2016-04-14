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


#ifdef DEBUG_PME_GPU

#include "thread_mpi/mutex.h"
#include "gromacs/gpu_utils/cudautils.cuh"

const bool check_verbose = false;
static tMPI::mutex print_mutex;

// prints differences in GPU/CPU values
template <typename T> void check(const char *name, T *data, T *expected, int size, gmx_bool bDevice, gmx_bool bPrintGrid = false)
{
    gmx_bool print1CharOnly = bPrintGrid;
    gmx_bool printEqualsAlso = print1CharOnly; //|=
    gmx_bool beganToPrintDiff = false;
    print_mutex.lock();
    for (int i = 0; i < size; ++i) 
    {
        T cpu_v = expected[i];
        T gpu_v;
        if (bDevice) 
        {
            cudaError_t stat = cudaMemcpy(&gpu_v, &data[i], sizeof(T), cudaMemcpyDeviceToHost);
            CU_RET_ERR(stat, "cudaMemcpy check error");
        }
        else 
            gpu_v = data[i];
        T diff = gpu_v - cpu_v;
        if (check_verbose) 
          fprintf(stderr, " %d:%f(%f)", i, (double) cpu_v, (double) diff);
        if (diff != 0) 
        {
            if (!beganToPrintDiff && !print1CharOnly)
            {
                if (name)
                    fprintf(stderr, "%s:\n", name);
                beganToPrintDiff = true;
            }
            T absdiff = diff > 0 ? diff : -diff;
            T abscpu_v = cpu_v > 0 ? cpu_v : -cpu_v;
            T reldiff = absdiff / (abscpu_v > 1e-11 ? abscpu_v : 1e-11);
            if (reldiff > 1e-6)
            {
                if (print1CharOnly)
                    fprintf(stderr, "&");
                else
                    fprintf(stderr, "%.0fppm ", (double) (reldiff * 1e6));
                if (reldiff > 1e-4)
                    if (print1CharOnly)
                        fprintf(stderr, "!");
                    else
                        fprintf(stderr, " value %f vs %f ", (double) cpu_v, (double) gpu_v);
            }
            else if (printEqualsAlso)
                fprintf(stderr, "~");
        }
        else if (printEqualsAlso)
        {
            if (gpu_v == 0)
            {
                fprintf(stderr, "0");
            }
            else
            {
                fprintf(stderr, "=");
            }
        }
    }
    if (beganToPrintDiff || printEqualsAlso)
        fprintf(stderr, "\n");
    print_mutex.unlock();
}

void check_int(const char *name, int *data, int *expected, int size, gmx_bool bDevice, gmx_bool bPrintGrid)
{
  check(name, data, expected, size, bDevice, bPrintGrid);
}

void check_real(const char *name, real *data, real *expected, int size, gmx_bool bDevice, gmx_bool bPrintGrid)
{
  check(name, data, expected, size, bDevice, bPrintGrid);
}

void print_lock() {
  print_mutex.lock();
}

void print_unlock() {
  print_mutex.lock();
}
#endif
