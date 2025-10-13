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
 * Tests for functionality of the "dump" tool.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 */
#include "gmxpre.h"

#include "gromacs/tools/dump.h"

#include "gromacs/gmxpreprocess/grompp.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/cmdlinetest.h"
#include "testutils/testfilemanager.h"
#include "testutils/tprfilegenerator.h"

namespace gmx
{

namespace test
{

class DumpTest : public ::testing::Test
{
public:
    //! Run test case.
    static void runTest(CommandLine* cmdline);

protected:
    // TODO this is changed in newer googletest versions
    //! Prepare shared resources.
    static void SetUpTestSuite() { s_tprFileHandle = new TprAndFileManager("lysozyme"); }
    //! Clean up shared resources.
    static void TearDownTestSuite()
    {
        delete s_tprFileHandle;
        s_tprFileHandle = nullptr;
    }
    //! Storage for opened file handles.
    static TprAndFileManager* s_tprFileHandle;
};

TprAndFileManager* DumpTest::s_tprFileHandle = nullptr;

void DumpTest::runTest(CommandLine* cmdline)
{
    EXPECT_EQ(0, gmx::test::CommandLineTestHelper::runModuleFactory(&gmx::DumpInfo::create, cmdline));
}

TEST_F(DumpTest, WorksWithTpr)
{
    const char* const command[] = { "dump", "-s", s_tprFileHandle->tprName().c_str() };
    CommandLine       cmdline(command);
    runTest(&cmdline);
}

TEST_F(DumpTest, WorksWithTprAndMdpWriting)
{
    TestFileManager fileManager;
    std::string     mdpName = fileManager.getTemporaryFilePath("output.mdp").u8string();
    const char* const command[] = { "dump", "-s", s_tprFileHandle->tprName().c_str(), "-om", mdpName.c_str() };
    CommandLine cmdline(command);
    runTest(&cmdline);
}

} // namespace test

} // namespace gmx
