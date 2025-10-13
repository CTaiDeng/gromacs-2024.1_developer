/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Tests for genconf.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 */

#include "gmxpre.h"

#include "gromacs/gmxpreprocess/genconf.h"

#include "testutils/cmdlinetest.h"
#include "testutils/refdata.h"
#include "testutils/testfilemanager.h"
#include "testutils/textblockmatchers.h"

namespace
{

using gmx::test::CommandLine;
using gmx::test::ExactTextMatch;

class GenconfTest : public gmx::test::CommandLineTestBase
{
public:
    GenconfTest()
    {
        std::string confFileName =
                gmx::test::TestFileManager::getInputFilePath("spc-and-methanol.gro").u8string();
        commandLine().addOption("-f", confFileName);
        commandLine().addOption("-seed", "1993"); // make random operations reproducible
        setOutputFile("-o", "out.gro", ExactTextMatch());
    }

    void runTest(const CommandLine& args)
    {
        CommandLine& cmdline = commandLine();
        cmdline.merge(args);

        gmx::test::TestReferenceChecker rootChecker(this->rootChecker());
        rootChecker.checkString(args.toString(), "CommandLine");

        ASSERT_EQ(0, gmx_genconf(cmdline.argc(), cmdline.argv()));

        checkOutputFiles();
    }
};

TEST_F(GenconfTest, nbox_Works)
{
    const char* const cmdline[] = { "genconf", "-nbox", "2", "1", "1" };
    runTest(CommandLine(cmdline));
}

TEST_F(GenconfTest, nbox_norenumber_Works)
{
    const char* const cmdline[] = { "genconf", "-nbox", "2", "1", "1", "-norenumber" };
    runTest(CommandLine(cmdline));
}

TEST_F(GenconfTest, nbox_dist_Works)
{
    const char* const cmdline[] = { "genconf", "-nbox", "2", "2", "3", "-dist", "0.1" };
    runTest(CommandLine(cmdline));
}

TEST_F(GenconfTest, nbox_rot_Works)
{
    const char* const cmdline[] = { "genconf", "-nbox", "2", "2", "3", "-rot" };
    runTest(CommandLine(cmdline));
}

} // namespace
