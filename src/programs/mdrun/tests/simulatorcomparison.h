/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * Helper classes for tests that compare the results of equivalent
 * simulation runs. Currently used for the rerun and the simulator
 * tests
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrun_integration_tests
 */

#ifndef GMX_PROGRAMS_MDRUN_TESTS_SIMULATORCOMPARISON_H
#define GMX_PROGRAMS_MDRUN_TESTS_SIMULATORCOMPARISON_H

#include <string>

#include "comparison_helpers.h"
#include "energycomparison.h"
#include "trajectorycomparison.h"

namespace gmx
{
namespace test
{
class SimulationRunner;

typedef std::tuple<std::string, std::string> SimulationOptionTuple;

void runGrompp(SimulationRunner*                         runner,
               const std::vector<SimulationOptionTuple>& options = std::vector<SimulationOptionTuple>());

void runMdrun(SimulationRunner*                         runner,
              const std::vector<SimulationOptionTuple>& options = std::vector<SimulationOptionTuple>());

void compareEnergies(const std::string&          edr1Name,
                     const std::string&          edr2Name,
                     const EnergyTermsToCompare& energyTermsToCompare,
                     MaxNumFrames                maxNumFrams = MaxNumFrames::compareAllFrames());

void compareTrajectories(const std::string&          trajectory1Name,
                         const std::string&          trajectory2Name,
                         const TrajectoryComparison& trajectoryComparison);

} // namespace test
} // namespace gmx

#endif // GMX_PROGRAMS_MDRUN_TESTS_SIMULATORCOMPARISON_H
