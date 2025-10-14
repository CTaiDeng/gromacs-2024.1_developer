/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2022- The GROMACS Authors
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
 * Tests for functionality of the "dssp" trajectory analysis module.
 *
 * \author Sergey Gorelov <gorelov_sv@pnpi.nrcki.ru>
 * \author Anatoly Titov <titov_ai@pnpi.nrcki.ru>
 * \author Alexey Shvetsov <alexxyum@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/modules/dssp.h"

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

/********************************************************************
 * Tests for gmx::analysismodules::Dssp.
 */

using DsspTestParamsDsspNB      = std::tuple<std::string, real, std::string, std::string>;
using DsspTestParamsGromacsNB   = std::tuple<std::string, real, std::string, std::string>;
using DsspTestParamsDsspNoNB    = std::tuple<std::string, std::string, std::string>;
using DsspTestParamsGromacsNoNB = std::tuple<std::string, std::string, std::string>;

//! Test fixture for the dssp analysis module.
class DsspModuleTestDsspNB :
    public TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::DsspInfo>,
    public ::testing::WithParamInterface<DsspTestParamsDsspNB>
{
};

//! Test fixture for the dssp analysis module.
class DsspModuleTestGromacsNB :
    public TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::DsspInfo>,
    public ::testing::WithParamInterface<DsspTestParamsGromacsNB>
{
};
//! Test fixture for the dssp analysis module.
class DsspModuleTestDsspNoNB :
    public TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::DsspInfo>,
    public ::testing::WithParamInterface<DsspTestParamsDsspNoNB>
{
};

//! Test fixture for the dssp analysis module.
class DsspModuleTestGromacsNoNB :
    public TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::DsspInfo>,
    public ::testing::WithParamInterface<DsspTestParamsGromacsNoNB>
{
};

// See https://github.com/google/googletest/issues/2442 for the reason
// for this and following NOLINTNEXTLINE suppressions.

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(DsspModuleTestDsspNB, Works)
{
    std::tuple<std::string, real, std::string, std::string> params    = GetParam();
    const char* const                                       cmdline[] = { "dssp" };
    std::string                                             inputFilename(std::get<0>(params));
    std::filesystem::path inputBasename = stripExtension(inputFilename);
    CommandLine           command(cmdline);
    setTopology(inputFilename.c_str());
    setTrajectory(inputFilename.c_str());
    setOutputFile("-o",
                  formatString("%s-dssp-nb-%.1f-%s-%s.dat",
                               inputBasename.c_str(),
                               std::get<1>(params),
                               std::get<2>(params).c_str(),
                               std::get<3>(params).c_str())
                          .c_str(),
                  ExactTextMatch());
    command.addOption("-hmode", "dssp");
    command.addOption("-nb");
    command.addOption("-cutoff", std::get<1>(params));
    command.addOption("-hbond", std::get<2>(params));
    command.addOption(std::string("-" + std::get<3>(params)).c_str());
    setOutputFile("-num",
                  formatString("%s-dssp-nb-%.1f-%s-%s.xvg",
                               inputBasename.c_str(),
                               std::get<1>(params),
                               std::get<2>(params).c_str(),
                               std::get<3>(params).c_str())
                          .c_str(),
                  test::XvgMatch());
    runTest(command);
}

INSTANTIATE_TEST_SUITE_P(MoleculeTests,
                         DsspModuleTestDsspNB,
                         ::testing::Combine(::testing::Values("1cos.pdb",
                                                              "1hlc.pdb",
                                                              "1vzj.pdb",
                                                              "3byc.pdb",
                                                              "3kyy.pdb",
                                                              "4r80.pdb",
                                                              "4xjf.pdb",
                                                              "5u5p.pdb",
                                                              "7wgh.pdb",
                                                              "1gmc.pdb",
                                                              "1v3y.pdb",
                                                              "1yiw.pdb",
                                                              "2os3.pdb",
                                                              "3u04.pdb",
                                                              "4r6c.pdb",
                                                              "4wxl.pdb",
                                                              "5cvq.pdb",
                                                              "5i2b.pdb",
                                                              "5t8z.pdb",
                                                              "6jet.pdb"),
                                            ::testing::Values(0.9, 2.0),
                                            ::testing::Values("energy", "geometry"),
                                            ::testing::Values("clear", "noclear")));


// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(DsspModuleTestGromacsNB, Works)
{
    std::tuple<std::string, real, std::string, std::string> params    = GetParam();
    const char* const                                       cmdline[] = { "dssp" };
    std::string                                             inputFilename(std::get<0>(params));
    std::filesystem::path inputBasename = stripExtension(inputFilename);
    CommandLine           command(cmdline);
    setTopology(inputFilename.c_str());
    setTrajectory(inputFilename.c_str());
    setOutputFile("-o",
                  formatString("%s-gromacs-nb-%.1f-%s-%s.dat",
                               inputBasename.c_str(),
                               std::get<1>(params),
                               std::get<2>(params).c_str(),
                               std::get<3>(params).c_str())
                          .c_str(),
                  ExactTextMatch());
    command.addOption("-hmode", "gromacs");
    command.addOption("-nb");
    command.addOption("-cutoff", std::get<1>(params));
    command.addOption("-hbond", std::get<2>(params));
    command.addOption(std::string("-" + std::get<3>(params)).c_str());
    setOutputFile("-num",
                  formatString("%s-gromacs-nb-%.1f-%s-%s.xvg",
                               inputBasename.c_str(),
                               std::get<1>(params),
                               std::get<2>(params).c_str(),
                               std::get<3>(params).c_str())
                          .c_str(),
                  test::XvgMatch());
    runTest(command);
}

INSTANTIATE_TEST_SUITE_P(
        MoleculeTests,
        DsspModuleTestGromacsNB,
        ::testing::Combine(::testing::Values("hdac.pdb", "RNAseA.pdb", "zyncfinger.pdb"),
                           ::testing::Values(0.9, 2.0),
                           ::testing::Values("energy", "geometry"),
                           ::testing::Values("clear", "noclear")));

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(DsspModuleTestDsspNoNB, Works)
{
    const char* const                                 cmdline[]     = { "dssp" };
    std::tuple<std::string, std::string, std::string> params        = GetParam();
    std::string                                       inputFilename = std::get<0>(params);
    std::filesystem::path                             inputBasename = stripExtension(inputFilename);
    CommandLine                                       command(cmdline);
    setTopology(inputFilename.c_str());
    setTrajectory(inputFilename.c_str());
    setOutputFile("-o",
                  formatString("%s-dssp-nonb-%s-%s.dat",
                               inputBasename.c_str(),
                               std::get<1>(params).c_str(),
                               std::get<2>(params).c_str())
                          .c_str(),
                  ExactTextMatch());
    command.addOption("-hmode", "dssp");
    command.addOption("-nonb");
    command.addOption("-hbond", std::get<1>(params));
    command.addOption(std::string("-" + std::get<2>(params)).c_str());
    setOutputFile("-num",
                  formatString("%s-dssp-nonb-%s-%s.xvg",
                               inputBasename.c_str(),
                               std::get<1>(params).c_str(),
                               std::get<2>(params).c_str())
                          .c_str(),
                  test::XvgMatch());
    runTest(command);
}

INSTANTIATE_TEST_SUITE_P(MoleculeTests,
                         DsspModuleTestDsspNoNB,
                         ::testing::Combine(::testing::Values("1cos.pdb",
                                                              "1hlc.pdb",
                                                              "1vzj.pdb",
                                                              "3byc.pdb",
                                                              "3kyy.pdb",
                                                              "4r80.pdb",
                                                              "4xjf.pdb",
                                                              "5u5p.pdb",
                                                              "7wgh.pdb",
                                                              "1gmc.pdb",
                                                              "1v3y.pdb",
                                                              "1yiw.pdb",
                                                              "2os3.pdb",
                                                              "3u04.pdb",
                                                              "4r6c.pdb",
                                                              "4wxl.pdb",
                                                              "5cvq.pdb",
                                                              "5i2b.pdb",
                                                              "5t8z.pdb",
                                                              "6jet.pdb"),
                                            ::testing::Values("energy", "geometry"),
                                            ::testing::Values("clear", "noclear")));


// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(DsspModuleTestGromacsNoNB, Works)
{
    const char* const                                 cmdline[]     = { "dssp" };
    std::tuple<std::string, std::string, std::string> params        = GetParam();
    std::string                                       inputFilename = std::get<0>(params);
    std::filesystem::path                             inputBasename = stripExtension(inputFilename);
    CommandLine                                       command(cmdline);
    setTopology(inputFilename.c_str());
    setTrajectory(inputFilename.c_str());
    setOutputFile("-o",
                  formatString("%s-gromacs-nonb-%s-%s.dat",
                               inputBasename.c_str(),
                               std::get<1>(params).c_str(),
                               std::get<2>(params).c_str())
                          .c_str(),
                  ExactTextMatch());
    command.addOption("-hmode", "gromacs");
    command.addOption("-nonb");
    command.addOption("-hbond", std::get<1>(params));
    command.addOption(std::string("-" + std::get<2>(params)).c_str());
    setOutputFile("-num",
                  formatString("%s-gromacs-nonb-%s-%s.xvg",
                               inputBasename.c_str(),
                               std::get<1>(params).c_str(),
                               std::get<2>(params).c_str())
                          .c_str(),
                  test::XvgMatch());
    runTest(command);
}

INSTANTIATE_TEST_SUITE_P(
        MoleculeTests,
        DsspModuleTestGromacsNoNB,
        ::testing::Combine(::testing::Values("hdac.pdb", "RNAseA.pdb", "zyncfinger.pdb"),
                           ::testing::Values("energy", "geometry"),
                           ::testing::Values("clear", "noclear")));

} // namespace
} // namespace test
} // namespace gmx
