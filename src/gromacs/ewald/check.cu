#include <cuda.h>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

#ifdef DEBUG_PME_TIMINGS_GPU

#include "gromacs/timing/gpu_timing.h"
#include "gromacs/timing/wallcycle.h"


struct gpu_events
{
  bool created;
  cudaEvent_t event_start, event_stop;
  gpu_events() : created(false) { }
};

gpu_events gpu_events_interpol_idx;
gpu_events gpu_events_calcspline;
gpu_events gpu_events_spread;
gpu_events gpu_events_fft_r2c;
gpu_events gpu_events_solve;
gpu_events gpu_events_fft_c2r;
gpu_events gpu_events_gather;

void events_record_start(gpu_events &events)
{
  if (!events.created) {
    cudaEventCreate(&events.event_start);
    cudaEventCreate(&events.event_stop);
    events.created = true;
  }
  cudaEventRecord(events.event_start);
}

void events_record_stop(gpu_events &events, int ewcsn, int j) {
  cudaEventRecord(events.event_stop);
  cudaEventSynchronize(events.event_stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, events.event_start, events.event_stop);

  int idx = ewcsn - ewcsPME_INTERPOL_IDX;
  gmx_wallclock_gpu_pme.pme_time[idx][j].t += milliseconds;
  ++gmx_wallclock_gpu_pme.pme_time[idx][j].c;
}
#endif



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