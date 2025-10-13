/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 * Tests for gmx genrestr.
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 */

#include "gmxpre.h"

#include "gromacs/gmxpreprocess/genrestr.h"

#include "testutils/cmdlinetest.h"
#include "testutils/filematchers.h"
#include "testutils/refdata.h"
#include "testutils/stdiohelper.h"
#include "testutils/testfilemanager.h"
#include "testutils/textblockmatchers.h"

namespace gmx
{
namespace test
{

class GenRestrTest : public CommandLineTestBase
{
public:
    void runTest(const std::string& interactiveCommandLineInput)
    {
        StdioTestHelper stdIoHelper(&fileManager());
        stdIoHelper.redirectStringToStdin(interactiveCommandLineInput.c_str());

        CommandLine& cmdline = commandLine();
        // Provide the name of the module to call
        std::string module[] = { "genrestr" };
        cmdline.merge(CommandLine(module));

        ASSERT_EQ(0, gmx_genrestr(cmdline.argc(), cmdline.argv()));
        checkOutputFiles();
    }
};

TEST_F(GenRestrTest, SimpleRestraintsGenerated)
{
    setInputFile("-f", "lysozyme.pdb");
    ExactTextMatch settings;
    setOutputFile("-o", "restraints.itp", TextFileMatch(settings));
    // Select c-alphas from default index options.
    std::string selection = "3";
    runTest(selection);
}
} // namespace test
} // namespace gmx
