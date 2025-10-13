/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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
 * Tests for the mdrun multi-simulation functionality
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "config.h"

#include <gtest/gtest.h>

#include "gromacs/mdtypes/md_enums.h"

#include "multisimtest.h"

namespace gmx
{
namespace test
{

/* This test ensures mdrun can run multi-simulations.  It runs one
 * simulation per MPI rank.
 *
 * TODO Preferably, we could test that mdrun correctly refuses to run
 * multi-simulation unless compiled with real MPI with more than one
 * rank available. However, if we just call mdrun blindly, those cases
 * trigger an error that is currently fatal to mdrun and also to the
 * test binary. So, in the meantime we must not test those cases. If
 * there is no MPI, we disable the test, so that there is a reminder
 * that it is disabled. There's no elegant way to conditionally
 * disable a test at run time, so currently there is no feedback if
 * only one rank is available. However, the test harness knows to run
 * this test with more than one rank. */
TEST_P(MultiSimTest, ExitsNormally)
{
    runExitsNormallyTest();
}

TEST_P(MultiSimTest, ExitsNormallyWithDifferentNumbersOfStepsPerSimulation)
{
    if (!mpiSetupValid())
    {
        // MPI setup is not suitable for multi-sim
        return;
    }
    SimulationRunner runner(&fileManager_);
    runner.useTopGroAndNdxFromDatabase("spc2");

    // Do some different small numbers of steps in each simulation
    int numSteps = simulationNumber_ % 4;
    runGrompp(&runner, numSteps);

    ASSERT_EQ(0, runner.callMdrun(*mdrunCaller_));
}

/* Note, not all preprocessor implementations nest macro expansions
   the same way / at all, if we would try to duplicate less code. */
#if GMX_LIB_MPI
INSTANTIATE_TEST_SUITE_P(InNvt,
                         MultiSimTest,
                         ::testing::Combine(::testing::Values(NumRanksPerSimulation(1),
                                                              NumRanksPerSimulation(2)),
                                            ::testing::Values(IntegrationAlgorithm::MD),
                                            ::testing::Values(TemperatureCoupling::VRescale),
                                            ::testing::Values(PressureCoupling::No)));
#else
// Test needs real MPI to run
INSTANTIATE_TEST_SUITE_P(DISABLED_InNvt,
                         MultiSimTest,
                         ::testing::Combine(::testing::Values(NumRanksPerSimulation(1),
                                                              NumRanksPerSimulation(2)),
                                            ::testing::Values(IntegrationAlgorithm::MD),
                                            ::testing::Values(TemperatureCoupling::VRescale),
                                            ::testing::Values(PressureCoupling::No)));
#endif

//! Convenience typedef
typedef MultiSimTest MultiSimTerminationTest;

TEST_P(MultiSimTerminationTest, WritesCheckpointAfterMaxhTerminationAndThenRestarts)
{
    runMaxhTest();
}

INSTANTIATE_TEST_SUITE_P(InNvt,
                         MultiSimTerminationTest,
                         ::testing::Combine(::testing::Values(NumRanksPerSimulation(1),
                                                              NumRanksPerSimulation(2)),
                                            ::testing::Values(IntegrationAlgorithm::MD),
                                            ::testing::Values(TemperatureCoupling::VRescale),
                                            ::testing::Values(PressureCoupling::No)));

} // namespace test
} // namespace gmx
