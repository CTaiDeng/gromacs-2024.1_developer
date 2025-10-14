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
 * Tests for gmx traj.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/gmxana/gmx_ana.h"

#include "testutils/cmdlinetest.h"
#include "testutils/simulationdatabase.h"
#include "testutils/stdiohelper.h"
#include "testutils/textblockmatchers.h"
#include "testutils/xvgtest.h"

namespace
{

class GmxTraj : public gmx::test::CommandLineTestBase, public ::testing::WithParamInterface<const char*>
{
public:
    void runTest(const char* fileName)
    {
        auto& cmdline = commandLine();
        setInputFile("-s", "spc2.gro");
        setInputFile("-f", fileName);
        setOutputFile("-ox", "spc2.xvg", gmx::test::NoTextMatch());

        gmx::test::StdioTestHelper stdioHelper(&fileManager());
        stdioHelper.redirectStringToStdin("0\n");

        ASSERT_EQ(0, gmx_traj(cmdline.argc(), cmdline.argv()));
        checkOutputFiles();
    }
};

/* TODO These tests are actually not very effective, because gmx-traj
 * can only return 0 or exit via gmx_fatal() (which currently also
 * exits the test binary). So, "no XDR/TNG support in the binary"
 * leads to the reading test appearing to pass. Still, no fatal error
 * and no segfault is useful information while modifying such code. */

TEST_P(GmxTraj, WithDifferentInputFormats)
{
    runTest(GetParam());
}

TEST_P(GmxTraj, RotationalKineticEnergy)
{
    setOutputFile("-ekr",
                  "spc2_ekr.xvg",
                  gmx::test::XvgMatch().tolerance(gmx::test::relativeToleranceAsFloatingPoint(1.0, 1e-4)));
    runTest(GetParam());
}

/*! \brief Helper array of input files present in the source repo
 * database. These all have two identical frames of two SPC water
 * molecules, which were generated via trjconv from the .gro
 * version. */
const char* const trajectoryFileNames[] = { "spc2-traj.trr", "spc2-traj.xtc", "spc2-traj.gro",
                                            "spc2-traj.pdb", "spc2-traj.g96",
#if GMX_USE_TNG
                                            "spc2-traj.tng"
#endif
};

INSTANTIATE_TEST_SUITE_P(NoFatalErrorWhenWritingFrom, GmxTraj, ::testing::ValuesIn(trajectoryFileNames));

} // namespace
