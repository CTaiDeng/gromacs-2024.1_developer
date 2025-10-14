/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Tests for general functionality in gmx::TrajectoryAnalysisCommandLineRunner.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/cmdlinerunner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/commandline/cmdlinemodule.h"
#include "gromacs/commandline/cmdlineoptionsmodule.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysismodule.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/exceptions.h"

#include "testutils/cmdlinetest.h"
#include "testutils/testasserts.h"

namespace
{

class MockModule : public gmx::TrajectoryAnalysisModule
{
public:
    MOCK_METHOD2(initOptions,
                 void(gmx::IOptionsContainer* options, gmx::TrajectoryAnalysisSettings* settings));
    MOCK_METHOD2(initAnalysis,
                 void(const gmx::TrajectoryAnalysisSettings& settings, const gmx::TopologyInformation& top));

    MOCK_METHOD4(analyzeFrame,
                 void(int frnr, const t_trxframe& fr, t_pbc* pbc, gmx::TrajectoryAnalysisModuleData* pdata));
    MOCK_METHOD1(finishAnalysis, void(int nframes));
    MOCK_METHOD0(writeOutput, void());
};

using gmx::test::CommandLine;

class TrajectoryAnalysisCommandLineRunnerTest : public gmx::test::CommandLineTestBase
{
public:
    TrajectoryAnalysisCommandLineRunnerTest() : mockModule_(new MockModule()) {}

    gmx::ICommandLineOptionsModulePointer createRunner()
    {
        return gmx::TrajectoryAnalysisCommandLineRunner::createModule(std::move(mockModule_));
    }

    void runTest(const CommandLine& args)
    {
        CommandLine& cmdline = commandLine();
        cmdline.merge(args);
        ASSERT_EQ(0, gmx::test::CommandLineTestHelper::runModuleDirect(createRunner(), &cmdline));
    }

    std::unique_ptr<MockModule> mockModule_;
};

//! Initializes options for help testing.
void initOptions(gmx::IOptionsContainer* options, gmx::TrajectoryAnalysisSettings* settings)
{
    const char* const desc[] = { "Sample description", "for testing [THISMODULE]." };
    settings->setHelpText(desc);

    options->addOption(gmx::BooleanOption("test").description("Test option"));
}

TEST_F(TrajectoryAnalysisCommandLineRunnerTest, WritesHelp)
{
    using ::testing::_;
    using ::testing::Invoke;
    EXPECT_CALL(*mockModule_, initOptions(_, _)).WillOnce(Invoke(&initOptions));

    const std::unique_ptr<gmx::ICommandLineModule> module(
            gmx::ICommandLineOptionsModule::createModule("mod", "Description", createRunner()));
    testWriteHelp(module.get());
}

TEST_F(TrajectoryAnalysisCommandLineRunnerTest, RunsWithSubsetTrajectory)
{
    const char* const cmdline[] = { "-fgroup", "atomnr 4 5 6 10 to 14" };

    using ::testing::_;
    EXPECT_CALL(*mockModule_, initOptions(_, _));
    EXPECT_CALL(*mockModule_, initAnalysis(_, _));
    EXPECT_CALL(*mockModule_, analyzeFrame(0, _, _, _));
    EXPECT_CALL(*mockModule_, analyzeFrame(1, _, _, _));
    EXPECT_CALL(*mockModule_, finishAnalysis(2));
    EXPECT_CALL(*mockModule_, writeOutput());

    setInputFile("-s", "simple.gro");
    setInputFile("-f", "simple-subset.gro");
    EXPECT_NO_THROW_GMX(runTest(CommandLine(cmdline)));
}

TEST_F(TrajectoryAnalysisCommandLineRunnerTest, DetectsIncorrectTrajectorySubset)
{
    const char* const cmdline[] = { "-fgroup", "atomnr 3 to 6 10 to 14" };

    using ::testing::_;
    EXPECT_CALL(*mockModule_, initOptions(_, _));
    EXPECT_CALL(*mockModule_, initAnalysis(_, _));

    setInputFile("-s", "simple.gro");
    setInputFile("-f", "simple-subset.gro");
    EXPECT_THROW_GMX(runTest(CommandLine(cmdline)), gmx::InconsistentInputError);
}

TEST_F(TrajectoryAnalysisCommandLineRunnerTest, FailsWithTrajectorySubsetWithoutTrajectory)
{
    const char* const cmdline[] = { "-fgroup", "atomnr 3 to 6 10 to 14" };

    using ::testing::_;
    EXPECT_CALL(*mockModule_, initOptions(_, _));

    setInputFile("-s", "simple.gro");
    EXPECT_THROW_GMX(runTest(CommandLine(cmdline)), gmx::InconsistentInputError);
}

} // namespace
