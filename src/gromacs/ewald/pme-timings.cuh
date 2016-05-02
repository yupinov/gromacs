#ifndef PME_TIMINGS_CUH
#define PME_TIMINGS_CUH

#include "pme-internal.h"

class pme_gpu_timing
{
    bool initialized;
    cudaEvent_t event_start, event_stop;
    unsigned int call_count;
    real total_milliseconds;

    void check_init();
public:
    pme_gpu_timing();
    ~pme_gpu_timing();

    void start_recording(cudaStream_t s);
    void stop_recording(cudaStream_t s);

    void update();
    void reset();

    real get_total_time_milliseconds();
    unsigned int get_call_count();
};

void pme_gpu_timing_start(gmx_pme_t *pme, int ewcsn);
void pme_gpu_timing_stop(gmx_pme_t *pme, int ewcsn);
void pme_gpu_update_timing(gmx_pme_t *pme); //internal!
void pme_gpu_get_timing(gmx_pme_t *pme);

#endif
