#ifndef PME_TIMINGS_CUH
#define PME_TIMINGS_CUH

#include "pme-internal.h"
#include "gromacs/timing/gpu_timing.h"

class pme_gpu_timing
{
    bool initialized;
    cudaEvent_t event_start, event_stop;
    unsigned int call_count;
    real total_milliseconds;

public:
    pme_gpu_timing();
    ~pme_gpu_timing();

    // to be called every MD step if needed
    void start_recording(cudaStream_t s);
    void stop_recording(cudaStream_t s);
    void update();

    // to be called once if needed
    void enable();
    void reset();
    real get_total_time_milliseconds();
    unsigned int get_call_count();
};

void pme_gpu_init_timings(gmx_pme_t *pme);
void pme_gpu_timing_start(gmx_pme_t *pme, int PMEStageId);
void pme_gpu_timing_stop(gmx_pme_t *pme, int PMEStageId);
void pme_gpu_update_timings(gmx_pme_t *pme);
void pme_gpu_destroy_timings(gmx_pme_t *pme);

#endif
