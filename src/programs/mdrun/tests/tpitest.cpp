/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
 * Copyright (C) 2025- GaoZheng
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * ---
 *
 * This file is part of a modified version of the GROMACS molecular simulation package.
 * For details on the original project, consult https://www.gromacs.org.
 *
 * To help fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

/*! \internal \file
 * \brief
 * This implements basic TPI sanity test.
 * It runs the input system with TPI for several steps, and checks the log output.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textreader.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{
namespace
{

/*! \brief A basic TPI runner.
 * The only input parameter used currently: input system random seed (ld-seed).
 */
class TpiTest : public MdrunTestFixture, public ::testing::WithParamInterface<int>
{
public:
    //! Runs the test with the given inputs
    void runTest();
};

void TpiTest::runTest()
{
    runner_.useTopGroAndNdxFromDatabase("spc216_with_methane");
    runner_.ndxFileName_ = "";
    ASSERT_EQ(0, runner_.callGrompp());

    auto        rerunFileName = gmx::test::TestFileManager::getInputFilePath("spc216.gro");
    CommandLine commandLine;
    commandLine.append("-rerun");
    commandLine.append(rerunFileName.u8string());
    ASSERT_EQ(0, runner_.callMdrun(commandLine));

    const std::string logFileContexts = TextReader::readFileToString(runner_.logFileName_);
    const std::string tpiOutputs =
            logFileContexts.substr(logFileContexts.find("Started Test Particle Insertion"));

    TestReferenceData    refData;
    TestReferenceChecker checker(refData.rootChecker());

    // Output values, their output patterns and relative tolerances (empirical)
    const std::map<std::string, std::pair<std::string, double>> valuesToCheck = {
        { "V", { "<V>  =", 1e-10 } }, { "mu", { "<mu> =", 1e-3 } }
    };

    for (const auto& valueDesc : valuesToCheck)
    {
        const auto& name              = valueDesc.first;
        const auto& pattern           = valueDesc.second.first;
        const auto& relativeTolerance = valueDesc.second.second;
        auto        startIndex        = tpiOutputs.find(pattern);
        ASSERT_NE(startIndex, std::string::npos);
        startIndex += pattern.size();
        const double actualValue = std::stod(tpiOutputs.substr(startIndex));
        checker.setDefaultTolerance(relativeToleranceAsFloatingPoint(actualValue, relativeTolerance));
        checker.checkDouble(actualValue, name.c_str());
    }
}

TEST_P(TpiTest, ReproducesOutput)
{
    const int randomSeed = GetParam();

    const int         nsteps          = 200;
    const std::string mdpFileContents = formatString(R"(
        integrator               = tpi
        ld-seed                  = %d
        rtpi                     = 0.2
        nstlog                   = 0
        nstenergy                = 0
        cutoff-scheme            = Verlet
        nstlist                  = 10
        rlist                    = 0.9
        coulombtype              = reaction-field
        rcoulomb                 = 0.9
        epsilon-r                = 1
        epsilon-rf               = 0
        vdw-type                 = cut-off
        vdw-modifier             = none
        rvdw                     = 0.9
        Tcoupl                   = no
        tc-grps                  = System
        tau_t                    = 0.5
        ref_t                    = 298
        nsteps                   = %d
    )",
                                                     randomSeed,
                                                     nsteps);

    runner_.useStringAsMdpFile(mdpFileContents);
    runTest();
}

INSTANTIATE_TEST_SUITE_P(Simple, TpiTest, ::testing::Values(1993, 2994));

} // namespace
} // namespace test
} // namespace gmx
