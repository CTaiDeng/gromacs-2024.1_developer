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
 * Tests for functionality of the "convert-trj" trajectory analysis module.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/modules/convert_trj.h"

#include <gtest/gtest.h>

#include "testutils/cmdlinetest.h"
#include "testutils/filematchers.h"
#include "testutils/textblockmatchers.h"

#include "moduletest.h"

namespace
{

using gmx::test::CommandLine;
using gmx::test::ExactTextMatch;
using gmx::test::NoContentsMatch;

/********************************************************************
 * Tests for gmx::analysismodules::ConvertTrj.
 */

//! Test fixture for the convert-trj analysis module.
typedef gmx::test::TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::ConvertTrjInfo> ConvertTrjModuleTest;

TEST_F(ConvertTrjModuleTest, WritesNormalOutput)
{
    const char* const cmdline[] = { "convert-trj" };
    setTopology("freevolume.tpr");
    setInputFile("-f", "freevolume.xtc");
    setOutputFile("-o", "test.trr", NoContentsMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(ConvertTrjModuleTest, WritesAtomSubset)
{
    const char* const cmdline[] = { "convert-trj", "-select", "not resname = CO2" };
    setTopology("freevolume.tpr");
    setInputFile("-f", "freevolume.xtc");
    setOutputFile("-o", "test.trr", NoContentsMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(ConvertTrjModuleTest, WorksWithAtomAdding)
{
    const char* const cmdline[] = { "convert-trj", "-atoms", "always-from-structure" };
    // TODO check output structures once this is supported.
    setTopology("clustsize.tpr");
    setInputFile("-f", "clustsize.pdb");
    setOutputFile("-o", "test.gro", ExactTextMatch());
    runTest(CommandLine(cmdline));
}

TEST_F(ConvertTrjModuleTest, WorksWithAtomsAndSelection)
{
    const char* const cmdline[] = {
        "convert-trj", "-atoms", "always-from-structure", "-select", "not resname = CO2"
    };
    // TODO check output structures once this is supported.
    setTopology("clustsize.tpr");
    setInputFile("-f", "clustsize.pdb");
    setOutputFile("-o", "test.gro", ExactTextMatch());
    runTest(CommandLine(cmdline));
}

} // namespace
