#ifndef PME_TIMINGS_CUH
#define PME_TIMINGS_CUH

#include "pme-internal.h"

struct pme_gpu_timing
{
    bool created;
    cudaEvent_t event_start, event_stop;
    pme_gpu_timing() : created(false) { }
};

void pme_gpu_timing_start(gmx_pme_t *pme, int ewcsn);
void pme_gpu_timing_stop(gmx_pme_t *pme, int ewcsn);
void pme_gpu_timing_calculate(gmx_pme_t *pme);

#endif
