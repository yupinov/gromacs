//This is a top-level file to generate all PME openCL kernels

//FIXME #define CUSTOMIZED_KERNEL_NAME(x) x for CUDA

/* SPREAD/SPLINE */

#define atomsPerBlock 16 //FIXME si having this here better than renaming it?


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


#undef atomsPerBlock


/* GATHER */

#define atomsPerBlock 8


// gather
#define overwriteForces 1
#define CUSTOMIZED_KERNEL_NAME(x) pmeGatherKernel
#include "../../ewald/pme-gather-kernel.cl"
#undef overwriteForces
#undef CUSTOMIZED_KERNEL_NAME

// gather with reduction
#define overwriteForces 0
#define CUSTOMIZED_KERNEL_NAME(x) pmeGatherReduceWithInputKernel
#include "../../ewald/pme-gather-kernel.cl"
#undef overwriteForces
#undef CUSTOMIZED_KERNEL_NAME


#undef atomsPerBlock
