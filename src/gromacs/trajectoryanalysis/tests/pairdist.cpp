/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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
 * Tests for functionality of the "pairdist" trajectory analysis module.
 *
 * These tests test the basic functionality of the tool itself, but currently
 * the following are missing:
 *  - Tests related to -odg output.  This would require a full tpr file, and
 *    some investigation on what kind of tpr it should be to produce reasonable
 *    output.
 *  - Tests for the X axes in the area per atom/residue plots.  These could be
 *    added once better X axes are implemented.
 *  - Tests for XVG labels.  This is a limitation of the current testing
 *    framework.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/modules/pairdist.h"

#include <gtest/gtest.h>

#include "testutils/cmdlinetest.h"
#include "testutils/testasserts.h"
#include "testutils/textblockmatchers.h"

#include "moduletest.h"

namespace
{

using gmx::test::CommandLine;
using gmx::test::NoTextMatch;

/********************************************************************
 * Tests for gmx::analysismodules::PairDistance.
 */

//! Test fixture for the select analysis module.
typedef gmx::test::TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::PairDistanceInfo> PairDistanceModuleTest;

TEST_F(PairDistanceModuleTest, ComputesAllDistances)
{
    const char* const cmdline[] = { "pairdist",     "-ref",         "resindex 1",
                                    "-refgrouping", "none",         "-sel",
                                    "resindex 3",   "-selgrouping", "none" };
    setTopology("simple.gro");
    setOutputFile("-o", ".xvg", NoTextMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(PairDistanceModuleTest, ComputesAllDistancesWithCutoff)
{
    const char* const cmdline[] = { "pairdist", "-ref",    "resindex 1", "-refgrouping",
                                    "none",     "-sel",    "resindex 3", "-selgrouping",
                                    "none",     "-cutoff", "1.5" };
    setTopology("simple.gro");
    setOutputFile("-o", ".xvg", NoTextMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(PairDistanceModuleTest, ComputesMinDistanceWithCutoff)
{
    const char* const cmdline[] = { "pairdist",   "-ref",    "resindex 1", "-sel",
                                    "resindex 3", "-cutoff", "1.5" };
    setTopology("simple.gro");
    setOutputFile("-o", ".xvg", NoTextMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(PairDistanceModuleTest, ComputesMaxDistance)
{
    const char* const cmdline[] = { "pairdist",   "-ref",  "resindex 1", "-sel",
                                    "resindex 3", "-type", "max" };
    setTopology("simple.gro");
    setOutputFile("-o", ".xvg", NoTextMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(PairDistanceModuleTest, ComputesMaxDistanceWithCutoff)
{
    const char* const cmdline[] = { "pairdist", "-ref", "resindex 1", "-sel", "resindex 3",
                                    "-cutoff",  "1.5",  "-type",      "max" };
    setTopology("simple.gro");
    setOutputFile("-o", ".xvg", NoTextMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(PairDistanceModuleTest, ComputesGroupedMinDistanceWithCutoff)
{
    const char* const cmdline[] = { "pairdist",        "-ref",         "resindex 1 to 2",
                                    "-refgrouping",    "res",          "-sel",
                                    "resindex 3 to 5", "-selgrouping", "res",
                                    "-cutoff",         "2.5" };
    setTopology("simple.gro");
    setOutputFile("-o", ".xvg", NoTextMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(PairDistanceModuleTest, ComputesGroupedMaxDistanceWithCutoff)
{
    const char* const cmdline[] = { "pairdist",
                                    "-ref",
                                    "resindex 1 to 2",
                                    "-refgrouping",
                                    "res",
                                    "-sel",
                                    "resindex 3 to 5",
                                    "-selgrouping",
                                    "res",
                                    "-cutoff",
                                    "3.5",
                                    "-type",
                                    "max" };
    setTopology("simple.gro");
    setOutputFile("-o", ".xvg", NoTextMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(PairDistanceModuleTest, CoordinateSelectionIsNotOverwritten)
{
    const char* const cmdline[] = { "pairdist", "-ref", "[0.0, 1.5, 2.9]", "-sel", "resindex 3",
                                    "-type",    "max" };
    setTopology("simple.gro");
    setOutputFile("-o", ".xvg", NoTextMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(PairDistanceModuleTest, CoordinateSelectionIsNotOverwrittenWithExplicitGroup)
{
    const char* const cmdline[] = { "pairdist", "-ref", "[0.0, 1.5, 2.9]", "-sel", "resindex 3",
                                    "-type",    "max",  "-refgrouping",    "res" };
    setTopology("simple.gro");
    setOutputFile("-o", ".xvg", NoTextMatch());
    runTest(CommandLine(cmdline));
}

} // namespace
