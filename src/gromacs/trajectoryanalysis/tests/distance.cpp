/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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
 * Tests for functionality of the "distance" trajectory analysis module.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/modules/distance.h"

#include <gtest/gtest.h>

#include "testutils/cmdlinetest.h"

#include "moduletest.h"

namespace
{

using gmx::test::CommandLine;

/********************************************************************
 * Tests for gmx::analysismodules::Distance.
 */

//! Test fixture for the angle analysis module.
typedef gmx::test::TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::DistanceInfo> DistanceModuleTest;

TEST_F(DistanceModuleTest, ComputesDistances)
{
    const char* const cmdline[] = { "distance", "-select", "atomname S1 S2", "-len", "2",
                                    "-binw",    "0.5" };
    setTopology("simple.gro");
    runTest(CommandLine(cmdline));
}

TEST_F(DistanceModuleTest, ComputesMultipleDistances)
{
    const char* const cmdline[] = {
        "distance",       "-select",
        "atomname S1 S2", "resindex 1 to 4 and atomname CB merge resindex 2 to 5 and atomname CB",
        "-len",           "2",
        "-binw",          "0.5"
    };
    setTopology("simple.gro");
    runTest(CommandLine(cmdline));
}

TEST_F(DistanceModuleTest, HandlesDynamicSelections)
{
    const char* const cmdline[] = { "distance", "-select", "atomname S1 S2 and res_cog x < 2.8",
                                    "-len",     "2",       "-binw",
                                    "0.5" };
    setTopology("simple.gro");
    runTest(CommandLine(cmdline));
}

TEST_F(DistanceModuleTest, HandlesSelectionFromGroup)
{
    const char* const cmdline[] = { "distance", "-select", "group \"Contacts\"" };
    setInputFile("-n", "simple.ndx");
    setTopology("simple.gro");
    runTest(CommandLine(cmdline));
}

TEST_F(DistanceModuleTest, HandlesSelectionFromGroupWithSuccessiveIndices)
{
    // Ensure that the presence of repeated indices like "1 2 2 3" works
    const char* const cmdline[] = { "distance", "-select", "group \"SuccessiveContacts\"" };
    setInputFile("-n", "simple.ndx");
    setTopology("simple.gro");
    runTest(CommandLine(cmdline));
}

TEST_F(DistanceModuleTest, HandlesSelectionFromLargeGroup)
{
    const char* const cmdline[] = { "distance", "-select", "group \"ManyContacts\"" };
    setInputFile("-n", "simple.ndx");
    setTopology("simple.gro");
    runTest(CommandLine(cmdline));
}

} // namespace
