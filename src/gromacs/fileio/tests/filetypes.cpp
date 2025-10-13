/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2022- The GROMACS Authors
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
 * Tests for routines for file type handling.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_fileio
 */
#include "gmxpre.h"

#include "gromacs/fileio/filetypes.h"

#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/utility/path.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{
namespace test
{
namespace
{

using TypeAndName = std::tuple<int, std::string>;

using FileTypeTestParams = std::tuple<TypeAndName, std::string>;

class FileTypeTest : public ::testing::Test, public ::testing::WithParamInterface<FileTypeTestParams>
{
public:
    static void runTest(const TypeAndName& params);
};

void FileTypeTest::runTest(const TypeAndName& params)
{
    const int   type = std::get<0>(params);
    const auto& path = std::get<1>(params);
    EXPECT_EQ(type, fn2ftp(path));
    // also test
}

TEST_P(FileTypeTest, CorrectValueForNullptr)
{
    // A lot of places in the code still call fn2ftp(nullptr).
    ASSERT_EQ(fn2ftp(nullptr), efNR);
}

TEST_F(FileTypeTest, CorrectValueForEmptyString)
{
    runTest({ efNR, "" });
}

TEST_F(FileTypeTest, CorrectValueForNoExtension)
{
    runTest({ efNR, "test" });
}

TEST_F(FileTypeTest, CorrectValueForEmptyExtension)
{
    runTest({ efNR, "test." });
}

TEST_F(FileTypeTest, CorrectValueForLongExtensionWithStrangeCharacters)
{
    runTest({ efNR, "test.1234\\abc" });
}

TEST_P(FileTypeTest, CorrectValueForExtension)
{
    auto param       = GetParam();
    auto typeAndName = std::get<0>(param);
    auto prefix      = std::get<1>(param);
    auto fullName    = prefix + std::get<1>(typeAndName);
    runTest({ std::get<0>(typeAndName), fullName });
}

const std::vector<TypeAndName> testParams = {
    { 0, ".mdp" },  { 4, ".trr" },  { 6, ".xtc" },  { 7, ".tng" },  { 8, ".edr" },  { 11, ".gro" },
    { 12, ".g96" }, { 13, ".pdb" }, { 14, ".brk" }, { 15, ".ent" }, { 16, ".esp" }, { 17, ".pqr" },
    { 18, ".cpt" }, { 19, ".log" }, { 20, ".xvg" }, { 21, ".out" }, { 22, ".ndx" }, { 23, ".top" },
    { 24, ".itp" }, { 26, ".tpr" }, { 27, ".tex" }, { 28, ".rtp" }, { 29, ".atp" }, { 30, ".hdb" },
    { 31, ".dat" }, { 32, ".dlg" }, { 33, ".map" }, { 34, ".eps" }, { 35, ".mat" }, { 36, ".m2p" },
    { 37, ".mtx" }, { 38, ".edi" }, { 39, ".cub" }, { 40, ".xpm" }, { 42, ".csv" }, { 43, ".inp" }
};

const std::vector<std::string> prefixes = { "",
                                            "test",
                                            "test.pdb",
                                            "a/../b/test",
                                            "james.gro/system.mdp/test.pdb" };

INSTANTIATE_TEST_SUITE_P(FileTypeMatch,
                         FileTypeTest,
                         ::testing::Combine(::testing::ValuesIn(testParams), ::testing::ValuesIn(prefixes)));


} // namespace
} // namespace test
} // namespace gmx
