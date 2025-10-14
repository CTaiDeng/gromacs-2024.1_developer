/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * Tests for gmx clustsize
 *
 * \todo These will be superseded by tests of the new style analysis
 * modules.
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_trajectoryanalysis
 */

#include "gmxpre.h"

#include <string>

#include "gromacs/gmxana/gmx_ana.h"

#include "testutils/cmdlinetest.h"
#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "testutils/textblockmatchers.h"
#include "testutils/xvgtest.h"

namespace gmx
{

namespace test
{

namespace
{

class ClustsizeTest : public CommandLineTestBase
{
public:
    ClustsizeTest()
    {
        double         tolerance = 1e-4;
        test::XvgMatch xvg;
        test::XvgMatch& toler = xvg.tolerance(gmx::test::relativeToleranceAsFloatingPoint(1, tolerance));

        setOutputFile("-mc", ".xvg", toler);
        setOutputFile("-nc", ".xvg", toler);
        setOutputFile("-ac", ".xvg", toler);
        setOutputFile("-hc", ".xvg", toler);
        setInputFile("-f", "clustsize.pdb");
    }

    void runTest(const CommandLine& args)
    {
        CommandLine& cmdline = commandLine();
        cmdline.merge(args);

        gmx::test::TestReferenceChecker rootChecker(this->rootChecker());
        rootChecker.checkString(args.toString(), "CommandLine");

        ASSERT_EQ(0, gmx_clustsize(cmdline.argc(), cmdline.argv()));

        checkOutputFiles();
    }
};

TEST_F(ClustsizeTest, NoMolDefaultCutoff)
{
    const char* const command[] = { "clustsize" };
    CommandLine       args      = CommandLine(command);

    setInputFile("-n", "clustsize.ndx");

    runTest(args);
}

TEST_F(ClustsizeTest, NoMolShortCutoff)
{
    const char* const command[] = { "clustsize", "-cut", "0.3" };
    CommandLine       args      = CommandLine(command);

    setInputFile("-n", "clustsize.ndx");

    runTest(args);
}

TEST_F(ClustsizeTest, MolDefaultCutoff)
{
    const char* const command[] = { "clustsize", "-mol" };
    CommandLine       args      = CommandLine(command);

    setInputFile("-s", "clustsize.tpr");

    runTest(args);
}

TEST_F(ClustsizeTest, MolShortCutoff)
{
    const char* const command[] = { "clustsize", "-mol", "-cut", "0.3" };
    CommandLine       args      = CommandLine(command);

    setInputFile("-s", "clustsize.tpr");

    runTest(args);
}

TEST_F(ClustsizeTest, MolCSize)
{
    const char* const command[] = { "clustsize", "-mol", "-nlevels", "6" };
    CommandLine       args      = CommandLine(command);

    setOutputFile("-o", ".xpm", ExactTextMatch());
    setOutputFile("-ow", ".xpm", ExactTextMatch());

    setInputFile("-s", "clustsize.tpr");

    runTest(args);
}

} // namespace

} // namespace test

} // namespace gmx
