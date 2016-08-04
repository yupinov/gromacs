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

//! a basic PME GPU sanity test
class PMEGPUTest:
    public gmx::test::MdrunTestFixture,
    public ::testing::WithParamInterface<const char *>
{

};

/* Ensure grompp and mdrun run and potential energy converges in an expected way.*/
TEST_F(PMEGPUTest, ReproducesEnergies)
{
    std::string theMdpFile = gmx::formatString("coulombtype     = PME\n"
                                   "verlet-buffer-tolerance =-1\n"
                                   "rvdw            = 0.9\n"
                                   "rlist           = 0.9\n"
                                   "rcoulomb        = 0.9\n"
                                   //"fourierspacing  = 0.12\n"
                                   //"ewald-rtol      = 1e-5\n"
                                   "pme-order       = 4\n"
                                   "nsteps          = 20\n");

    runner_.useStringAsMdpFile(theMdpFile);

    const std::string inputFile = "spc216";
    runner_.useTopGroAndNdxFromDatabase(inputFile.c_str());

    // grompp should run without error
    EXPECT_EQ(0, runner_.callGrompp());

    std::vector<std::string> PMEModes;
    PMEModes.push_back("cpu");
    PMEModes.push_back("gpu");

    std::map<std::string, gmx::test::EnergyFrameReaderPtr> energyReadersByMode;

    for (auto &it: PMEModes)
    {
        runner_.edrFileName_ = fileManager_.getTemporaryFilePath(inputFile + "_" + it + ".edr");

        ::gmx::test::CommandLine PMECommandLine;
        //PMECommandLine.addOption("-ntmpi", 1); // already declared?
        PMECommandLine.addOption("-pme", it);
        //PMECommandLine.addOption("-notunepme", ...

        // assert that mdrun is finished without error
        ASSERT_EQ(0, runner_.callMdrun(PMECommandLine));

        energyReadersByMode[it] = gmx::test::openEnergyFileToReadFields(runner_.edrFileName_, {{"Potential"}});
    }
    /*
    gmx::test::TestReferenceData    data;
    gmx::test::TestReferenceChecker checker(data.rootChecker());


    checker.setDefaultTolerance(gmx::test::relativeToleranceAsFloatingPoint(1.0, 1e-6));
    */
    for (int i = 0; i < 20; i++) //runner_.nsteps
    {
        for (auto &it: PMEModes)
            energyReadersByMode[it]->readNextFrame(); // +1?

        gmx::test::compareFrames(std::make_pair(energyReadersByMode[PMEModes[0]]->frame(),
                                                energyReadersByMode[PMEModes[1]]->frame()),
                                 gmx::test::relativeToleranceAsUlp(1.0, 7));//defaultRealTolerance());
    }
}

#ifdef __INTEL_COMPILER
#pragma warning( disable : 177 )
#endif

} // namespace
