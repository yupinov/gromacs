/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016,2017, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Implements common routines for PME tests.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "pmetestcommon.h"

#include "config.h"

#include <cstring>

#include "gromacs/ewald/pme-gather.h"
#include "gromacs/ewald/pme-gpu-internal.h"
#include "gromacs/ewald/pme-grid.h"
#include "gromacs/ewald/pme-internal.h"
#include "gromacs/ewald/pme-solve.h"
#include "gromacs/ewald/pme-spread.h"
#include "gromacs/fft/parallel_3dfft.h"
#include "gromacs/math/invertmatrix.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{
namespace test
{

bool pmeSupportsInputForMode(const t_inputrec *inputRec, CodePath mode)
{
    bool implemented;
    switch (mode)
    {
        case CodePath::CPU:
            implemented = true;
            break;

        case CodePath::CUDA:
            implemented = (GMX_GPU == GMX_GPU_CUDA) && (inputRec->pme_order == 4);
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
    // This copies some of the logic from pme_gpu_check_restrictions().
    // Duplicating the constraints here emphasizes them (changing them requires changing the test as well).
    return implemented;
}

//! PME initialization - internal
static PmeSafePointer pmeInitInternal(const t_inputrec         *inputRec,
                                      CodePath                  mode,
                                      size_t                    atomCount,
                                      const Matrix3x3          &box,
                                      real                      ewaldCoeff_q = 1.0f,
                                      real                      ewaldCoeff_lj = 1.0f
                                      )
{
    gmx_pme_t           *pmeDataRaw   = nullptr;
    const bool           useGpu       = (mode == CodePath::CPU) ? false : true;
    const gmx_hw_info_t *hardwareInfo = pmeEnv->getHardwareInfo();
    const gmx_gpu_opt_t *gpuOptions   = pmeEnv->getGpuOptions();

    gmx_pme_init(&pmeDataRaw, nullptr, 1, 1, inputRec, atomCount, false, false, true,
                 ewaldCoeff_q, ewaldCoeff_lj, 1, useGpu, nullptr, hardwareInfo, gpuOptions);
    PmeSafePointer pme(pmeDataRaw); // taking ownership

    // TODO get rid of this with proper matrix type
    matrix boxTemp;
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            boxTemp[i][j] = box[i * DIM + j];
        }
    }
    invertBoxMatrix(boxTemp, pme->recipbox);

    return pme;
}

//! Simple PME initialization based on input, no atom data
PmeSafePointer pmeInitEmpty(const t_inputrec         *inputRec,
                            CodePath                  mode,
                            const Matrix3x3          &box,
                            real                      ewaldCoeff_q,
                            real                      ewaldCoeff_lj)
{
    return pmeInitInternal(inputRec, mode, 0, box, ewaldCoeff_q, ewaldCoeff_lj);
    // hiding the fact that PME actually needs to know the number of atoms in advance
}

//! PME initialization with atom data
PmeSafePointer pmeInitWithAtoms(const t_inputrec         *inputRec,
                                CodePath                  mode,
                                const CoordinatesVector  &coordinates,
                                const ChargesVector      &charges,
                                const Matrix3x3          &box
                                )
{
    const size_t    atomCount = coordinates.size();
    GMX_RELEASE_ASSERT(atomCount == charges.size(), "Mismatch in atom data");
    PmeSafePointer  pmeSafe = pmeInitInternal(inputRec, mode, atomCount, box);
    pme_atomcomm_t *atc     = nullptr;

    // TODO get rid of this with proper matrix type
    matrix boxTemp;
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            boxTemp[i][j] = box[i * DIM + j];
        }
    }

    switch (mode)
    {
        case CodePath::CPU:
            atc              = &(pmeSafe->atc[0]);
            atc->x           = const_cast<rvec *>(as_rvec_array(coordinates.data()));
            atc->coefficient = const_cast<real *>(charges.data());
            /* With decomposition there would be more boilerplate atc code here, e.g. do_redist_pos_coeffs */
            break;

        case CodePath::CUDA:
            pme_gpu_set_testing(pmeSafe->gpu, true);
            gmx_pme_reinit_atoms(pmeSafe.get(), atomCount, charges.data());
            pme_gpu_start_step(pmeSafe->gpu, boxTemp, as_rvec_array(coordinates.data()));
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }

    return pmeSafe;
}

//! Getting local PME real grid pointer for test I/O
static real *pmeGetRealGridInternal(const gmx_pme_t *pme)
{
    const size_t gridIndex = 0;
    return pme->fftgrid[gridIndex];
}

//! Getting local PME real grid dimensions
static void pmeGetRealGridSizesInternal(const gmx_pme_t      *pme,
                                        IVec                 &gridSize,
                                        IVec                 &paddedGridSize)
{
    const size_t gridIndex = 0;
    IVec         gridOffsetUnused;
    gmx_parallel_3dfft_real_limits(pme->pfft_setup[gridIndex], gridSize, gridOffsetUnused, paddedGridSize);
}

//! Getting local PME complex grid pointer for test I/O
static t_complex *pmeGetComplexGridInternal(const gmx_pme_t *pme)
{
    const size_t gridIndex = 0;
    return pme->cfftgrid[gridIndex];
}

//! Getting local PME complex grid dimensions
static void pmeGetComplexGridSizesInternal(const gmx_pme_t      *pme,
                                           IVec                 &gridSize,
                                           IVec                 &paddedGridSize)
{
    const size_t gridIndex = 0;
    IVec         gridOffsetUnused, complexOrderUnused;
    gmx_parallel_3dfft_complex_limits(pme->pfft_setup[gridIndex], complexOrderUnused, gridSize, gridOffsetUnused, paddedGridSize); //TODO: what about YZX ordering?
}

//! Getting the PME grid memory buffer and its sizes - template definition
template<typename ValueType> static void pmeGetGridAndSizesInternal(const gmx_pme_t *, ValueType * &, IVec &, IVec &)
{
    GMX_THROW(InternalError("Deleted function call"));
    // explicitly deleting general template does not compile in clang/icc, see https://llvm.org/bugs/show_bug.cgi?id=17537
}

//! Getting the PME real grid memory buffer and its sizes
template<> void pmeGetGridAndSizesInternal<real>(const gmx_pme_t *pme, real * &grid, IVec &gridSize, IVec &paddedGridSize)
{
    grid = pmeGetRealGridInternal(pme);
    pmeGetRealGridSizesInternal(pme, gridSize, paddedGridSize);
}

//! Getting the PME complex grid memory buffer and its sizes
template<> void pmeGetGridAndSizesInternal<t_complex>(const gmx_pme_t *pme, t_complex * &grid, IVec &gridSize, IVec &paddedGridSize)
{
    grid = pmeGetComplexGridInternal(pme);
    pmeGetComplexGridSizesInternal(pme, gridSize, paddedGridSize);
}

//! PME spline calculation and charge spreading
void pmePerformSplineAndSpread(gmx_pme_t *pme, CodePath mode, // TODO const qualifiers elsewhere
                               bool computeSplines, bool spreadCharges)
{
    GMX_RELEASE_ASSERT(pme != nullptr, "PME data is not initialized");
    pme_atomcomm_t *atc                          = &(pme->atc[0]);
    const size_t    gridIndex                    = 0;
    const bool      computeSplinesForZeroCharges = true;
    real           *fftgrid                      = spreadCharges ? pme->fftgrid[gridIndex] : nullptr;
    real           *h_grid                       = pme->pmegrid[gridIndex].grid.grid;

    switch (mode)
    {
        case CodePath::CPU:
            spread_on_grid(pme, atc, &pme->pmegrid[gridIndex], computeSplines, spreadCharges,
                           fftgrid, computeSplinesForZeroCharges, gridIndex);
            if (spreadCharges && !pme->bUseThreads)
            {
                wrap_periodic_pmegrid(pme, h_grid);
                copy_pmegrid_to_fftgrid(pme, h_grid, fftgrid, gridIndex);
            }
            break;

        case CodePath::CUDA:
            pme_gpu_spread(pme->gpu, gridIndex, h_grid, computeSplines, spreadCharges);
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
}

//! Getting the internal spline data buffer pointer
static real *pmeGetSplineDataInternal(const gmx_pme_t *pme, PmeSplineDataType type, int dimIndex)
{
    GMX_ASSERT((0 <= dimIndex) && (dimIndex < DIM), "Invalid dimension index");
    const pme_atomcomm_t *atc          = &(pme->atc[0]);
    const size_t          threadIndex  = 0;
    real                 *splineBuffer = nullptr;
    switch (type)
    {
        case PmeSplineDataType::Values:
            splineBuffer = atc->spline[threadIndex].theta[dimIndex];
            break;

        case PmeSplineDataType::Derivatives:
            splineBuffer = atc->spline[threadIndex].dtheta[dimIndex];
            break;

        default:
            GMX_THROW(InternalError("Unknown spline data type"));
    }
    return splineBuffer;
}

//! PME solving
void pmePerformSolve(const gmx_pme_t *pme, CodePath mode,
                     PmeSolveAlgorithm method, real cellVolume)
{
    t_complex      *grid                   = pmeGetComplexGridInternal(pme);
    const bool      computeEnergyAndVirial = true;
    const bool      useLorentzBerthelot    = false;
    const size_t    threadIndex            = 0;
    switch (mode)
    {
        case CodePath::CPU:
            switch (method)
            {
                case PmeSolveAlgorithm::Normal:
                    solve_pme_yzx(pme, grid, cellVolume,
                                  computeEnergyAndVirial, pme->nthread, threadIndex);
                    break;

                case PmeSolveAlgorithm::LennardJones:
                    solve_pme_lj_yzx(pme, &grid, useLorentzBerthelot,
                                     cellVolume, computeEnergyAndVirial, pme->nthread, threadIndex);
                    break;

                default:
                    GMX_THROW(InternalError("Test not implemented for this mode"));
            }
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
}

//! PME force gathering
void pmePerformGather(gmx_pme_t *pme, CodePath mode,
                      PmeGatherInputHandling inputTreatment, ForcesVector &forces)
{
    pme_atomcomm_t *atc                     = &(pme->atc[0]);
    const size_t    atomCount               = atc->n;
    GMX_RELEASE_ASSERT(forces.size() == atomCount, "Invalid force buffer size");
    const bool      forceReductionWithInput = (inputTreatment == PmeGatherInputHandling::ReduceWith);
    const real      scale                   = 1.0;
    const size_t    threadIndex             = 0;
    const size_t    gridIndex               = 0;
    real           *h_grid                  = pme->pmegrid[gridIndex].grid.grid;

    copy_fftgrid_to_pmegrid(pme, pme->fftgrid[gridIndex], h_grid, gridIndex, pme->nthread, threadIndex);

    switch (mode)
    {
        case CodePath::CPU:
            atc->f = as_rvec_array(forces.begin());
            if (atc->nthread == 1)
            {
                // something which is normally done in serial spline computation (make_thread_local_ind())
                atc->spline[threadIndex].n = atomCount;
            }
            unwrap_periodic_pmegrid(pme, h_grid);
            gather_f_bsplines(pme, h_grid, !forceReductionWithInput, atc, &atc->spline[threadIndex], scale);
            break;

        case CodePath::CUDA:
            pme_gpu_gather(pme->gpu, reinterpret_cast<float *>(forces.begin()), !forceReductionWithInput, reinterpret_cast<float *>(h_grid));
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
}

//! PME test finalization before fetching the outputs
void pmeFinalizeTest(const gmx_pme_t *pme, CodePath mode)
{
    switch (mode)
    {
        case CodePath::CPU:
            break;

        case CodePath::CUDA:
            pme_gpu_synchronize(pme->gpu);
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
}

//! Setting atom spline values/derivatives to be used in spread/gather
void pmeSetSplineData(const gmx_pme_t *pme, CodePath mode,
                      const SplineParamsDimVector &splineValues, PmeSplineDataType type, int dimIndex)
{
    const pme_atomcomm_t *atc         = &(pme->atc[0]);
    const size_t          atomCount   = atc->n;
    const size_t          pmeOrder    = pme->pme_order;
    const size_t          dimSize     = pmeOrder * atomCount;
    GMX_RELEASE_ASSERT(dimSize == splineValues.size(), "Mismatch in spline data");
    real                 *splineBuffer = pmeGetSplineDataInternal(pme, type, dimIndex);

    switch (mode)
    {
        case CodePath::CPU:
            std::copy(splineValues.begin(), splineValues.end(), splineBuffer);
            break;

        case CodePath::CUDA:
            std::copy(splineValues.begin(), splineValues.end(), splineBuffer);
            pme_gpu_transform_spline_atom_data(pme->gpu, atc, type, dimIndex, PmeLayoutTransform::HostToGpu);
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
}

//! Setting gridline indices to be used in spread/gather
void pmeSetGridLineIndices(const gmx_pme_t *pme, CodePath mode,
                           const GridLineIndicesVector &gridLineIndices)
{
    const pme_atomcomm_t       *atc         = &(pme->atc[0]);
    const size_t                atomCount   = atc->n;
    GMX_RELEASE_ASSERT(atomCount == gridLineIndices.size(), "Mismatch in gridline indices size");

    IVec paddedGridSizeUnused, gridSize;
    pmeGetRealGridSizesInternal(pme, gridSize, paddedGridSizeUnused);

    for (const auto &index : gridLineIndices)
    {
        for (int i = 0; i < DIM; i++)
        {
            GMX_RELEASE_ASSERT((0 <= index[i]) && (index[i] < gridSize[i]), "Invalid gridline index");
        }
    }

    switch (mode)
    {
        case CodePath::CUDA:
            memcpy(pme->gpu->staging.h_gridlineIndices, gridLineIndices.data(), atomCount * sizeof(gridLineIndices[0]));
            break;

        case CodePath::CPU:
            // incompatible IVec and ivec assignment?
            //std::copy(gridLineIndices.begin(), gridLineIndices.end(), atc->idx);
            memcpy(atc->idx, gridLineIndices.data(), atomCount * sizeof(gridLineIndices[0]));
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
}

//! Setting real or complex grid
template<typename ValueType>
static void pmeSetGridInternal(const gmx_pme_t *pme, CodePath mode,
                               const SparseGridValuesInput<ValueType> &gridValues)
{
    IVec       gridSize, paddedGridSize;
    ValueType *grid;
    pmeGetGridAndSizesInternal<ValueType>(pme, grid, gridSize, paddedGridSize);

    switch (mode)
    {
        case CodePath::CUDA: // intentional absence of break, the grid will be copied from the host buffer in testing mode
        case CodePath::CPU:
            std::memset(grid, 0, paddedGridSize[XX] * paddedGridSize[YY] * paddedGridSize[ZZ] * sizeof(ValueType));
            for (const auto &gridValue : gridValues)
            {
                for (int i = 0; i < DIM; i++)
                {
                    GMX_RELEASE_ASSERT((0 <= gridValue.first[i]) && (gridValue.first[i] < gridSize[i]), "Invalid grid value index");
                }
                const size_t gridValueIndex = (gridValue.first[XX] * paddedGridSize[YY] + gridValue.first[YY]) * paddedGridSize[ZZ] + gridValue.first[ZZ];
                grid[gridValueIndex] = gridValue.second;
            }
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
}

//! Setting real grid to be used in gather
void pmeSetRealGrid(const gmx_pme_t *pme, CodePath mode,
                    const SparseRealGridValuesInput &gridValues)
{
    pmeSetGridInternal<real>(pme, mode, gridValues);
}

//! Setting complex grid to be used in solve
void pmeSetComplexGrid(const gmx_pme_t *pme, CodePath mode,
                       const SparseComplexGridValuesInput &gridValues)
{
    pmeSetGridInternal<t_complex>(pme, mode, gridValues);
}

//! Getting the single dimension's spline values or derivatives
SplineParamsDimVector pmeGetSplineData(const gmx_pme_t *pme, CodePath mode,
                                       PmeSplineDataType type, int dimIndex)
{
    GMX_RELEASE_ASSERT(pme != nullptr, "PME data is not initialized");
    const pme_atomcomm_t    *atc         = &(pme->atc[0]);
    const size_t             atomCount   = atc->n;
    const size_t             pmeOrder    = pme->pme_order;
    const size_t             dimSize     = pmeOrder * atomCount;

    real                    *sourceBuffer = pmeGetSplineDataInternal(pme, type, dimIndex);
    SplineParamsDimVector    result;
    switch (mode)
    {
        case CodePath::CUDA:
            pme_gpu_transform_spline_atom_data(pme->gpu, atc, type, dimIndex, PmeLayoutTransform::GpuToHost);
        // intentional absence of break

        case CodePath::CPU:
            result = SplineParamsDimVector::fromArray(sourceBuffer, dimSize);
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
    return result;
}

//! Getting the gridline indices
GridLineIndicesVector pmeGetGridlineIndices(const gmx_pme_t *pme, CodePath mode)
{
    GMX_RELEASE_ASSERT(pme != nullptr, "PME data is not initialized");
    const pme_atomcomm_t *atc         = &(pme->atc[0]);
    const size_t          atomCount   = atc->n;

    GridLineIndicesVector gridLineIndices;
    switch (mode)
    {
        case CodePath::CUDA:
            gridLineIndices = GridLineIndicesVector::fromArray(reinterpret_cast<IVec *>(pme->gpu->staging.h_gridlineIndices), atomCount);
            break;

        case CodePath::CPU:
            gridLineIndices = GridLineIndicesVector::fromArray(reinterpret_cast<IVec *>(atc->idx), atomCount);
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
    return gridLineIndices;
}

//! Getting real or complex grid - only non zero values
template<typename ValueType>
static SparseGridValuesOutput<ValueType> pmeGetGridInternal(const gmx_pme_t *pme, CodePath mode)
{
    IVec       gridSize, paddedGridSize;
    ValueType *grid;
    pmeGetGridAndSizesInternal<ValueType>(pme, grid, gridSize, paddedGridSize);
    SparseGridValuesOutput<ValueType> gridValues;
    switch (mode)
    {
        case CodePath::CUDA: // intentional absence of break
        case CodePath::CPU:
            gridValues.clear();
            for (int ix = 0; ix < gridSize[XX]; ix++)
            {
                for (int iy = 0; iy < gridSize[YY]; iy++)
                {
                    for (int iz = 0; iz < gridSize[ZZ]; iz++)
                    {
                        const size_t    gridValueIndex = (ix * paddedGridSize[YY] + iy) * paddedGridSize[ZZ] + iz;
                        const ValueType value          = grid[gridValueIndex];
                        if (value != ValueType {})
                        {
                            auto key = formatString("Cell %d %d %d", ix, iy, iz);
                            gridValues[key] = value;
                        }
                    }
                }
            }
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
    return gridValues;
}

//! Getting the real grid (spreading output of pmePerformSplineAndSpread())
SparseRealGridValuesOutput pmeGetRealGrid(const gmx_pme_t *pme, CodePath mode)
{
    return pmeGetGridInternal<real>(pme, mode);
}

//! Getting the complex grid output of pmePerformSolve()
SparseComplexGridValuesOutput pmeGetComplexGrid(const gmx_pme_t *pme, CodePath mode)
{
    return pmeGetGridInternal<t_complex>(pme, mode);
}

//! Getting the reciprocal energy and virial
PmeSolveOutput pmeGetReciprocalEnergyAndVirial(const gmx_pme_t *pme, CodePath mode,
                                               PmeSolveAlgorithm method)
{
    real      energy = 0.0f;
    Matrix3x3 virial;
    matrix    virialTemp; //TODO get rid of
    switch (mode)
    {
        case CodePath::CPU:
            switch (method)
            {
                case PmeSolveAlgorithm::Normal:
                    get_pme_ener_vir_q(pme->solve_work, pme->nthread, &energy, virialTemp);
                    break;

                case PmeSolveAlgorithm::LennardJones:
                    get_pme_ener_vir_lj(pme->solve_work, pme->nthread, &energy, virialTemp);
                    break;

                default:
                    GMX_THROW(InternalError("Test not implemented for this mode"));
            }
            break;

        default:
            GMX_THROW(InternalError("Test not implemented for this mode"));
    }
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            virial[i * DIM + j] = virialTemp[i][j];
        }
    }
    return std::make_tuple(energy, virial);
}

}
}
