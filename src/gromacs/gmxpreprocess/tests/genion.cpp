/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * Tests for genion.
 *
 * \author Vytautas Gapsys <vgapsys@gwdg.de>
 * \author Christian Blau <blau@kth.se>
 */

#include "gmxpre.h"

#include "gromacs/gmxpreprocess/genion.h"

#include "gromacs/gmxpreprocess/grompp.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/textreader.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/cmdlinetest.h"
#include "testutils/conftest.h"
#include "testutils/refdata.h"
#include "testutils/stdiohelper.h"
#include "testutils/testfilemanager.h"
#include "testutils/textblockmatchers.h"


namespace gmx
{
namespace test
{
namespace
{

class GenionTest : public CommandLineTestBase
{
public:
    GenionTest()
    {
        CommandLine caller = commandLine();

        const std::string mdpInputFileName(fileManager().getTemporaryFilePath("input.mdp").u8string());
        TextWriter::writeFileFromString(
                mdpInputFileName,
                "verlet-buffer-tolerance =-1\nrcoulomb=0.5\nrvdw = 0.5\nrlist = 0.5\n");
        caller.addOption("-f", mdpInputFileName);
        caller.addOption("-c", TestFileManager::getInputFilePath("spc216_with_methane.gro").u8string());
        caller.addOption("-p", TestFileManager::getInputFilePath("spc216_with_methane.top").u8string());
        caller.addOption("-o", tprFileName_);

        gmx_grompp(caller.argc(), caller.argv());

        setOutputFile("-o", "out.gro", ConfMatch());
    }

    void runTest(const CommandLine& args, const std::string& interactiveCommandLineInput)
    {
        StdioTestHelper stdIoHelper(&fileManager());
        stdIoHelper.redirectStringToStdin(interactiveCommandLineInput.c_str());

        CommandLine& cmdline = commandLine();
        cmdline.addOption("-s", tprFileName_);
        cmdline.merge(args);

        ASSERT_EQ(0, gmx_genion(cmdline.argc(), cmdline.argv()));
        checkOutputFiles();
    }

private:
    const std::string tprFileName_ =
            fileManager().getTemporaryFilePath("spc216_with_methane.tpr").u8string();
};

TEST_F(GenionTest, HighConcentrationIonPlacement)
{
    const char* const cmdline[] = { "genion", "-seed", "1997", "-conc", "1.0", "-rmin", "0.6" };

    runTest(CommandLine(cmdline), "Water");
}

TEST_F(GenionTest, NoIonPlacement)
{
    const char* const cmdline[] = { "genion", "-seed", "1997", "-conc", "0.0", "-rmin", "0.6" };

    runTest(CommandLine(cmdline), "Water");
}

} // namespace
} // namespace test
} // namespace gmx
