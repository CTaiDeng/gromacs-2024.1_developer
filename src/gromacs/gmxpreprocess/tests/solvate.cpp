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
 * Tests for solvation.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 */

#include "gmxpre.h"

#include "gromacs/gmxpreprocess/solvate.h"

#include "gromacs/utility/futil.h"
#include "gromacs/utility/textreader.h"

#include "testutils/cmdlinetest.h"
#include "testutils/conftest.h"
#include "testutils/refdata.h"
#include "testutils/testfilemanager.h"
#include "testutils/textblockmatchers.h"

namespace
{

using gmx::test::CommandLine;
using gmx::test::ConfMatch;
using gmx::test::ExactTextMatch;

class SolvateTest : public gmx::test::CommandLineTestBase
{
public:
    SolvateTest() { setOutputFile("-o", "out.gro", ConfMatch()); }

    void runTest(const CommandLine& args)
    {
        CommandLine& cmdline = commandLine();
        cmdline.merge(args);

        ASSERT_EQ(0, gmx_solvate(cmdline.argc(), cmdline.argv()));
        checkOutputFiles();
    }
};

TEST_F(SolvateTest, cs_box_Works)
{
    // use default solvent box (-cs without argument)
    const char* const cmdline[] = { "solvate", "-cs", "-box", "1.1" };
    runTest(CommandLine(cmdline));
}

TEST_F(SolvateTest, cs_cp_Works)
{
    // use default solvent box (-cs without argument)
    const char* const cmdline[] = { "solvate", "-cs" };
    setInputFile("-cp", "spc-and-methanol.gro");
    runTest(CommandLine(cmdline));
}

TEST_F(SolvateTest, cs_cp_p_Works)
{
    // use default solvent box (-cs without argument)
    const char* const cmdline[] = { "solvate", "-cs" };
    setInputFile("-cp", "spc-and-methanol.gro");
    setModifiableInputFile("-p", "spc-and-methanol.top");

    runTest(CommandLine(cmdline));
}

TEST_F(SolvateTest, shell_Works)
{
    // use default solvent box (-cs without argument)
    const char* const cmdline[] = { "solvate", "-cs" };
    setInputFile("-cp", "spc-and-methanol.gro");
    commandLine().addOption("-shell", 1.0);

    runTest(CommandLine(cmdline));
}

TEST_F(SolvateTest, update_Topology_Works)
{
    // use solvent box with 2 solvents, check that topology has been updated
    const char* const cmdline[] = { "solvate" };
    setInputFile("-cs", "mixed_solvent.gro");
    setInputFile("-cp", "simple.gro");
    setInputAndOutputFile("-p", "simple.top", ExactTextMatch());

    runTest(CommandLine(cmdline));
}

TEST_F(SolvateTest, cs_pdb_big_box_Works)
{
    // use SPC216 solvent, but in PDB format
    const char* const cmdline[] = { "solvate", "-box", "2" };
    setInputFile("-cs", "spc216.pdb");
    runTest(CommandLine(cmdline));
}

} // namespace
