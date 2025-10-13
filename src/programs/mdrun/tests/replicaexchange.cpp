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
 * Tests for the mdrun replica-exchange functionality
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "config.h"

#include <regex>

#include <gtest/gtest.h>

#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/refdata.h"
#include "testutils/testfilemanager.h"

#include "energycomparison.h"
#include "multisimtest.h"
#include "trajectorycomparison.h"

namespace gmx
{
namespace test
{

//! Convenience typedef
typedef MultiSimTest ReplicaExchangeEnsembleTest;

TEST_P(ReplicaExchangeEnsembleTest, ExitsNormally)
{
    mdrunCaller_->addOption("-replex", 1);
    runExitsNormallyTest();
}

/* Note, not all preprocessor implementations nest macro expansions
   the same way / at all, if we would try to duplicate less code. */

#if GMX_LIB_MPI
INSTANTIATE_TEST_SUITE_P(
        WithDifferentControlVariables,
        ReplicaExchangeEnsembleTest,
        ::testing::Combine(::testing::Values(NumRanksPerSimulation(1), NumRanksPerSimulation(2)),
                           ::testing::Values(IntegrationAlgorithm::MD),
                           ::testing::Values(TemperatureCoupling::VRescale),
                           ::testing::Values(PressureCoupling::No, PressureCoupling::Berendsen)));
#else
INSTANTIATE_TEST_SUITE_P(
        DISABLED_WithDifferentControlVariables,
        ReplicaExchangeEnsembleTest,
        ::testing::Combine(::testing::Values(NumRanksPerSimulation(1), NumRanksPerSimulation(2)),
                           ::testing::Values(IntegrationAlgorithm::MD),
                           ::testing::Values(TemperatureCoupling::VRescale),
                           ::testing::Values(PressureCoupling::No, PressureCoupling::Berendsen)));
#endif

//! Convenience typedef
typedef MultiSimTest ReplicaExchangeTerminationTest;

TEST_P(ReplicaExchangeTerminationTest, WritesCheckpointAfterMaxhTerminationAndThenRestarts)
{
    mdrunCaller_->addOption("-replex", 1);
    runMaxhTest();
}

INSTANTIATE_TEST_SUITE_P(InNvt,
                         ReplicaExchangeTerminationTest,
                         ::testing::Combine(::testing::Values(NumRanksPerSimulation(1)),
                                            ::testing::Values(IntegrationAlgorithm::MD),
                                            ::testing::Values(TemperatureCoupling::VRescale),
                                            ::testing::Values(PressureCoupling::No)));

} // namespace test
} // namespace gmx
