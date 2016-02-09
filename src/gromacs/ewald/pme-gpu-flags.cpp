/*

  Some ad-hoc flags to control whether to run some code on cpu/gpu/both,
  to compare results, etc.

 */

#include "pme-internal.h"
#include "pme-spread.h"
#include "pme-gather.h"

struct gpu_flags
{
  enum run_on
  {
    CPU_ONLY, GPU_ONLY, BOTH, BOTH_CHECK, BOTH_CHECK_VERBOSE
  };
  run_on on;
  int j;
  gpu_flags(run_on on, int j = 0) : on(on), j(j) { }
};

bool run_on_cpu(const gpu_flags &flags) {
  return flags.on != gpu_flags::GPU_ONLY;
}

bool run_on_gpu(const gpu_flags &flags) {
  return flags.on != gpu_flags::CPU_ONLY;
}

bool check_vs_cpu(const gpu_flags &flags) {
  return flags.on >= gpu_flags::BOTH_CHECK;
}

bool check_vs_cpu_j(const gpu_flags &flags, int j) {
  return flags.on >= gpu_flags::BOTH_CHECK || flags.j == j;
}

bool check_vs_cpu_verbose(const gpu_flags &flags) {
  return flags.on >= gpu_flags::BOTH_CHECK_VERBOSE;
}

#define DEFAULT_FLAGS_PME gpu_flags::BOTH

gpu_flags interpol_gpu_flags(DEFAULT_FLAGS_PME);


gpu_flags calcspline_gpu_flags(DEFAULT_FLAGS_PME/*gpu_flags::BOTH_CHECK*/);
gpu_flags spread_gpu_flags(gpu_flags::BOTH_CHECK, -1); //?
gpu_flags spread_bunching_gpu_flags(DEFAULT_FLAGS_PME);
gpu_flags gather_gpu_flags(DEFAULT_FLAGS_PME);
gpu_flags fft_gpu_flags(DEFAULT_FLAGS_PME);
gpu_flags solve_gpu_flags(DEFAULT_FLAGS_PME);



bool check_solve_vs_cpu() { return true; }

