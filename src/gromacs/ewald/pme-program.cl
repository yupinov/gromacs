//This is a top-level file to generate all PME openCL kernels

//FIXME #define CUSTOMIZED_KERNEL_NAME(x) x for CUDA

// splineAndSpread
#define computeSplines 1
#define spreadCharges 1
#define CUSTOMIZED_KERNEL_NAME(x) pmeSplineAndSpreadKernel
#include "../../ewald/pme-spread-kernel.cl"
#undef computeSplines
#undef spreadCharges
#undef CUSTOMIZED_KERNEL_NAME

// spline
#define computeSplines 1
#define spreadCharges 0
#define CUSTOMIZED_KERNEL_NAME(x) pmeSplineKernel
#include "../../ewald/pme-spread-kernel.cl"
#undef computeSplines
#undef spreadCharges
#undef CUSTOMIZED_KERNEL_NAME

// spread
#define computeSplines 0
#define spreadCharges 1
#define CUSTOMIZED_KERNEL_NAME(x) pmeSpreadKernel
#include "../../ewald/pme-spread-kernel.cl"
#undef computeSplines
#undef spreadCharges
#undef CUSTOMIZED_KERNEL_NAME