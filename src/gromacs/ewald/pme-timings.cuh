#ifndef GMX_EWALD_CHECK_H
#define GMX_EWALD_CHECK_H

#ifdef DEBUG_PME_GPU

#include "gromacs/utility/basedefinitions.h"

struct gpu_flags;


bool run_on_cpu(const gpu_flags &flags);
bool run_on_gpu(const gpu_flags &flags);
bool check_vs_cpu(const gpu_flags &flags);
bool check_vs_cpu_j(const gpu_flags &flags, int j);
bool check_vs_cpu_verbose(const gpu_flags &flags);

void check_int(const char *name, int *data, int *expected, int size, gmx_bool bDevice, gmx_bool bPrintGrid = false);
void check_real(const char *name, real *data, real *expected, int size, gmx_bool bDevice, gmx_bool bPrintGrid = false);

void print_lock();
void print_unlock();
#endif


#include "gromacs/timing/wallcycle.h"
struct gpu_events
{
    bool created;
    cudaEvent_t event_start, event_stop;
    gpu_events() : created(false) { }
};
void events_record_start(gpu_events &events, cudaStream_t s);
void events_record_stop(gpu_events &events, cudaStream_t s, int ewcsn, int j);

extern gpu_events gpu_events_wrap, gpu_events_unwrap, gpu_events_gather;

#endif // GMX_EWALD_CHECK_H
