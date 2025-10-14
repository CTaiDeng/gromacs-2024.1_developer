/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
 * Copyright (C) 2025 GaoZheng
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
 * Tests  for scattering trajectory analysis module.
 *
 * \author Alexey Shvetsov <alexxyum@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/modules/scattering.h"

#include <string>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include "gromacs/utility/path.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/cmdlinetest.h"
#include "testutils/textblockmatchers.h"
#include "testutils/xvgtest.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{
namespace
{


using ScatteringTestDirectModeParams = std::tuple<std::string, std::string>;

//! Test fixture for the dssp analysis module.
class ScatteringModule :
    public TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::ScatteringInfo>,
    public ::testing::WithParamInterface<ScatteringTestDirectModeParams>
{
};


// See https://github.com/google/googletest/issues/2442 for the reason
// for this and following NOLINTNEXTLINE suppressions.

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(ScatteringModule, DirectMode)
{
    std::tuple<std::string, std::string> params    = GetParam();
    const char* const                    cmdline[] = { "scattering" };
    std::string                          inputFilename(std::get<0>(params));
    std::filesystem::path                inputBasename = stripExtension(inputFilename);
    CommandLine                          command(cmdline);
    double                               tolerance = 1e-3;
    test::XvgMatch                       xvg;
    test::XvgMatch& toler = xvg.tolerance(gmx::test::relativeToleranceAsFloatingPoint(1, tolerance));
    setTopology(inputFilename.c_str());
    setTrajectory(inputFilename.c_str());
    setOutputFile(
            "-o",
            formatString("%s-direct-%s.xvg", inputBasename.c_str(), std::get<1>(params).c_str()).c_str(),
            toler);
    command.addOption("-sel", "Protein");
    command.addOption("-scattering-type", std::get<1>(params));
    command.addOption("-norm");
    command.addOption("-nomc");
    setDatasetTolerance("scattering", gmx::test::relativeToleranceAsFloatingPoint(1, tolerance));
    runTest(command);
}

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(ScatteringModule, MCMode)
{
    std::tuple<std::string, std::string> params    = GetParam();
    const char* const                    cmdline[] = { "scattering" };
    std::string                          inputFilename(std::get<0>(params));
    std::filesystem::path                inputBasename = stripExtension(inputFilename);
    CommandLine                          command(cmdline);
    double                               tolerance = 1e-3;
    test::XvgMatch                       xvg;
    test::XvgMatch& toler = xvg.tolerance(gmx::test::relativeToleranceAsFloatingPoint(1, tolerance));
    setTopology(inputFilename.c_str());
    setTrajectory(inputFilename.c_str());
    setOutputFile("-o",
                  formatString("%s-mc-%s.xvg", inputBasename.c_str(), std::get<1>(params).c_str()).c_str(),
                  toler);
    command.addOption("-sel", "Protein");
    command.addOption("-scattering-type", std::get<1>(params));
    command.addOption("-norm");
    command.addOption("-mc");
    setDatasetTolerance("scattering", gmx::test::relativeToleranceAsFloatingPoint(1, tolerance));
    runTest(command);
}

INSTANTIATE_TEST_SUITE_P(MoleculeTests,
                         ScatteringModule,
                         ::testing::Combine(::testing::Values("lysozyme.pdb"),
                                            ::testing::Values("sans", "saxs")));


} // namespace
} // namespace test
} // namespace gmx
