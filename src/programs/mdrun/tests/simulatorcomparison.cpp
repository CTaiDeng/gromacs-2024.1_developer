/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * Helper functions for tests that compare the results of equivalent
 * simulation runs. Currently used for the rerun and the simulator
 * tests
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "simulatorcomparison.h"

#include "gromacs/trajectory/energyframe.h"

#include "testutils/trajectoryreader.h"

#include "energyreader.h"
#include "mdruncomparison.h"
#include "moduletest.h"

namespace gmx
{
namespace test
{

void runGrompp(SimulationRunner* runner, const std::vector<SimulationOptionTuple>& options)
{
    CommandLine caller;
    caller.append("grompp");

    for (const std::tuple<std::string, std::string>& option : options)
    {
        caller.addOption(std::get<0>(option).c_str(), std::get<1>(option));
    }

    EXPECT_EQ(0, runner->callGrompp(caller));
}

void runMdrun(SimulationRunner* runner, const std::vector<SimulationOptionTuple>& options)
{
    CommandLine caller;
    caller.append("mdrun");

    for (const std::tuple<std::string, std::string>& option : options)
    {
        caller.addOption(std::get<0>(option).c_str(), std::get<1>(option));
    }

    EXPECT_EQ(0, runner->callMdrun(caller));
}

void compareEnergies(const std::string&          edr1Name,
                     const std::string&          edr2Name,
                     const EnergyTermsToCompare& energyTermsToCompare,
                     const MaxNumFrames          maxNumFrames)
{
    // Build the functor that will compare energy frames on the chosen
    // energy terms.
    EnergyComparison energyComparison(energyTermsToCompare, maxNumFrames);

    // Build the manager that will present matching pairs of frames to compare.
    //
    // TODO Here is an unnecessary copy of keys (ie. the energy term
    // names), for convenience. In the future, use a range.
    auto                                namesOfEnergiesToMatch = energyComparison.getEnergyNames();
    FramePairManager<EnergyFrameReader> energyManager(
            openEnergyFileToReadTerms(edr1Name, namesOfEnergiesToMatch),
            openEnergyFileToReadTerms(edr2Name, namesOfEnergiesToMatch));
    // Compare the energy frames.
    energyManager.compareAllFramePairs<EnergyFrame>(energyComparison);
}

void compareTrajectories(const std::string&          trajectory1Name,
                         const std::string&          trajectory2Name,
                         const TrajectoryComparison& trajectoryComparison)
{
    // Build the manager that will present matching pairs of frames to compare
    FramePairManager<TrajectoryFrameReader> trajectoryManager(
            std::make_unique<TrajectoryFrameReader>(trajectory1Name),
            std::make_unique<TrajectoryFrameReader>(trajectory2Name));
    // Compare the trajectory frames.
    trajectoryManager.compareAllFramePairs<TrajectoryFrame>(trajectoryComparison);
}

} // namespace test
} // namespace gmx
