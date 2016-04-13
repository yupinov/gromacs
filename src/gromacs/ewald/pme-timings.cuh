#ifndef GMX_EWALD_CHECK_H
#define GMX_EWALD_CHECK_H

#include "pme-internal.h"

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


struct pme_gpu_timing
{
    bool created;
    cudaEvent_t event_start, event_stop;
    pme_gpu_timing() : created(false) { }
};

void pme_gpu_timing_start(gmx_pme_t *pme, int ewcsn);
void pme_gpu_timing_stop(gmx_pme_t *pme, int ewcsn);
void pme_gpu_timing_calculate(gmx_pme_t *pme);

#endif // GMX_EWALD_CHECK_H
