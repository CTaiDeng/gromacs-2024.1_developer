/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2023- The GROMACS Authors
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
 * Tests for functionality of the "gyrate" trajectory analysis module.
 *
 * \author Vladimir Basov <vovabasov830@gmail.com>
 * \author Alexey Shvetsov <alexxyum@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/modules/gyrate.h"

#include <string>
#include <tuple>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include "gromacs/utility/stringutil.h"

#include "testutils/cmdlinetest.h"
#include "testutils/textblockmatchers.h"
#include "testutils/xvgtest.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{

/********************************************************************
 * Tests for gmx::analysismodules::Gyrate.
 */

using GyrateTestParamsMerge = std::tuple<std::string, std::string>;

//! Test fixture for the gyrate analysis module.
class GyrateModuleTest :
    public TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::GyrateInfo>,
    public ::testing::WithParamInterface<GyrateTestParamsMerge>
{
};

// See https://github.com/google/googletest/issues/2442 for the reason
// for this and following NOLINTNEXTLINE suppressions.

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(GyrateModuleTest, Works)
{
    std::tuple<std::string, std::string> params        = GetParam();
    const char* const                    cmdline[]     = { "gyrate-taf" };
    std::string                          inputBasename = std::get<0>(params);
    CommandLine                          command(cmdline);
    double                               tolerance = 1e-5;
    test::XvgMatch                       matcher;
    test::XvgMatch&                      toleranceMatch =
            matcher.tolerance(gmx::test::relativeToleranceAsFloatingPoint(1, tolerance));
    setTrajectory((inputBasename + ".xtc").c_str());
    setTopology((inputBasename + ".tpr").c_str());
    command.addOption("-sel", "Protein");
    setOutputFile(
            "-o",
            formatString("%s-gyrate-%s.xvg", inputBasename.c_str(), std::get<1>(params).c_str()).c_str(),
            toleranceMatch);
    setDatasetTolerance("gyrate", gmx::test::relativeToleranceAsFloatingPoint(1, tolerance));
    runTest(command);
}

INSTANTIATE_TEST_SUITE_P(GyrateTests,
                         GyrateModuleTest,
                         ::testing::Combine(::testing::Values("trpcage"),
                                            ::testing::Values("mass", "charge", "geometry")));


} // namespace test
} // namespace gmx
