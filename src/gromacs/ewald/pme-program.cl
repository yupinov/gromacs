//This is a top-level file to generate all PME openCL kernels

// splineAndSpread
#define computeSplines 1
#define spreadCharges 1
#include "../../ewald/pme-spread-kernel.cl"
#undef computeSplines
#undef spreadCharges

// spline
#define computeSplines 1
#define spreadCharges 0
#include "../../ewald/pme-spread-kernel.cl"
#undef computeSplines
#undef spreadCharges

// spread
#define computeSplines 0
#define spreadCharges 1
#include "../../ewald/pme-spread-kernel.cl"
#undef computeSplines
#undef spreadCharges
