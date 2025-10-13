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
 * Tests for insertion of molecules.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 */
#include "gmxpre.h"

#include "gromacs/gmxpreprocess/insert_molecules.h"

#include "testutils/cmdlinetest.h"
#include "testutils/refdata.h"
#include "testutils/textblockmatchers.h"

namespace
{

using gmx::test::CommandLine;
using gmx::test::ExactTextMatch;

class InsertMoleculesTest : public gmx::test::CommandLineTestBase
{
public:
    InsertMoleculesTest() { setOutputFile("-o", "out.gro", ExactTextMatch()); }

    void runTest(const CommandLine& args)
    {
        CommandLine& cmdline = commandLine();
        cmdline.merge(args);

        gmx::test::TestReferenceChecker rootChecker(this->rootChecker());
        rootChecker.checkString(args.toString(), "CommandLine");

        ASSERT_EQ(0,
                  gmx::test::CommandLineTestHelper::runModuleFactory(
                          &gmx::InsertMoleculesInfo::create, &cmdline));

        checkOutputFiles();
    }
};

TEST_F(InsertMoleculesTest, InsertsMoleculesIntoExistingConfiguration)
{
    const char* const cmdline[] = { "insert-molecules", "-nmol", "1", "-seed", "1997" };
    setInputFile("-f", "spc-and-methanol.gro");
    setInputFile("-ci", "x2.gro");
    runTest(CommandLine(cmdline));
}

TEST_F(InsertMoleculesTest, InsertsMoleculesIntoEmptyBox)
{
    const char* const cmdline[] = { "insert-molecules", "-box", "4", "-nmol", "5", "-seed", "1997" };
    setInputFile("-ci", "x2.gro");
    runTest(CommandLine(cmdline));
}

TEST_F(InsertMoleculesTest, InsertsMoleculesIntoEnlargedBox)
{
    const char* const cmdline[] = { "insert-molecules", "-box", "4", "-nmol", "2", "-seed", "1997" };
    setInputFile("-f", "spc-and-methanol.gro");
    setInputFile("-ci", "x.gro");
    runTest(CommandLine(cmdline));
}

TEST_F(InsertMoleculesTest, InsertsMoleculesWithReplacement)
{
    const char* const cmdline[] = {
        "insert-molecules", "-nmol", "4", "-replace", "all", "-seed", "1997"
    };
    setInputFile("-f", "spc216.gro");
    setInputFile("-ci", "x.gro");
    runTest(CommandLine(cmdline));
}

TEST_F(InsertMoleculesTest, InsertsMoleculesIntoFixedPositions)
{
    const char* const cmdline[]   = { "insert-molecules", "-box", "4", "-seed", "1997" };
    const char* const positions[] = {
        "0.0  0.0  0.0", "1.0  2.0  3.0", "0.99 2.01 3.0", "2.0  1.0  2.0"
    };
    setInputFile("-ci", "x0.gro");
    setInputFileContents("-ip", "dat", positions);
    runTest(CommandLine(cmdline));
}

} // namespace
