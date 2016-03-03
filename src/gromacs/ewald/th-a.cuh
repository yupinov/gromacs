#ifndef GMX_EWALD_TH_A_H
#define GMX_EWALD_TH_A_H

#include "gromacs/utility/real.h"
#include "gromacs/math/gmxcomplex.h"

const int TH = 32;
enum th_id
{
  TH_ID_THETA, TH_ID_DTHETA, TH_ID_FRACTX, TH_ID_COEFFICIENT,

  TH_ID_GRID,
  TH_ID_I0, TH_ID_J0, TH_ID_K0,
  TH_ID_THX, TH_ID_THY, TH_ID_THZ,

  // interpol_idx
  TH_ID_G2T,
  TH_ID_FSH,
  TH_ID_NN,
  TH_ID_XPTR,

  TH_ID_IDXPTR, //yupinov added - a duplicate of TH_ID_I0, TH_ID_J0, TH_ID_K0,
  TH_ID_F,
  TH_ID_I,
  TH_ID_DTHX, TH_ID_DTHY, TH_ID_DTHZ,
  TH_ID_BSP_MOD_X, TH_ID_BSP_MOD_Y, TH_ID_BSP_MOD_Z,
  TH_ID_ENERGY,
  TH_ID_VIRIAL,

  TH_ID_END
};

enum th_loc
{
  TH_LOC_HOST, TH_LOC_CUDA, TH_LOC_END
};

real *th_a(th_id id, int thread, int size = -1, th_loc loc = TH_LOC_END);
int *th_i(th_id id, int thread, int size = -1, th_loc loc = TH_LOC_END);
t_complex *th_c(th_id id, int thread, int size = -1, th_loc loc = TH_LOC_END);

void th_cpy(void *dest, void *src, int size, th_loc dest_loc, cudaStream_t s); //yupinov alloc as well

int *th_i_cpy(th_id id, int thread, void *src, int size, th_loc loc, cudaStream_t s);
real *th_a_cpy(th_id id, int thread, void *src, int size, th_loc loc, cudaStream_t s);
t_complex *th_c_cpy(th_id id, int thread, void *src, int size, th_loc loc, cudaStream_t s);

//yupinov warn o nwrong param

#endif
