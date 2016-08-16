/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2016, by the GROMACS development team, led by
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
 * A basic PME GPU test
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/cmdlinetest.h" //yupinov


#include "testutils/refdata.h"

#include "energyreader.h"
#include "moduletest.h"

namespace
{

//! A basic PME GPU sanity test
class PMEGPUTest:
    public gmx::test::MdrunTestFixture,
    public ::testing::WithParamInterface<const char *>
{

};

/* Ensure 2 mdruns with CPU and GPU PME produce same reciprocal energies.*/
TEST_F(PMEGPUTest, ReproducesEnergies)
{
    int nsteps = 20;
    std::string theMdpFile = gmx::formatString("coulombtype             = PME\n"
                                               "nstcalcenergy   = 1\n"
                                               "nstenergy       = 1\n"
                                               "pme-order       = 4\n"
                                               "nsteps          = %d\n",
                                               nsteps);

    runner_.useStringAsMdpFile(theMdpFile);

    const std::string inputFile = "spc2";
    runner_.useTopGroAndNdxFromDatabase(inputFile.c_str());

    const real approximateCoulomb = 7.5; // Coulomb reciprocal energy value for spc2
    const real relativeTolerance = 1e-4;
    const gmx::test::FloatingPointTolerance tol = gmx::test::relativeToleranceAsFloatingPoint(approximateCoulomb, relativeTolerance);

    EXPECT_EQ(0, runner_.callGrompp());

    std::vector<std::string> PMEModes;
    PMEModes.push_back("cpu");
    PMEModes.push_back("gpu");

    std::map<std::string, gmx::test::EnergyFrameReaderPtr> energyReadersByMode;

    for (auto &it: PMEModes)
    {
        runner_.edrFileName_ = fileManager_.getTemporaryFilePath(inputFile + "_" + it + ".edr");

        ::gmx::test::CommandLine PMECommandLine;
        // PMECommandLine.addOption("-ntmpi", 1); /* already declared? */
        PMECommandLine.addOption("-pme", it);

        ASSERT_EQ(0, runner_.callMdrun(PMECommandLine));

        energyReadersByMode[it] = gmx::test::openEnergyFileToReadFields(runner_.edrFileName_, {{"Coul. recip."}}); /*, {"Conserved En."}});*/
    }

    for (int i = 0; i <= nsteps; i++)
    {
        for (auto &it: PMEModes)
        {
            energyReadersByMode[it]->readNextFrame();
        }

        gmx::test::compareFrames(std::make_pair(energyReadersByMode[PMEModes[0]]->frame(),
                                                energyReadersByMode[PMEModes[1]]->frame()),
                                 tol);
    }
}

#ifdef __INTEL_COMPILER
#pragma warning( disable : 177 )
#endif

} // namespace
