/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * Tests for (some) functions in path.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/path.h"

#include <utility>

#include <gtest/gtest.h>

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

TEST(PathTest, StripSourcePrefixWorks)
{
    EXPECT_EQ("", stripSourcePrefix(""));
    EXPECT_EQ("foo.cpp", stripSourcePrefix("foo.cpp"));
    EXPECT_EQ("foo.cpp", stripSourcePrefix("some/dir/foo.cpp"));
    EXPECT_EQ("foo.cpp", stripSourcePrefix("src/some/dir/foo.cpp"));
    EXPECT_EQ("foo.cpp", stripSourcePrefix("srcx/gromacs/foo.cpp"));
    EXPECT_EQ("src/gromacs/foo.cpp", stripSourcePrefix("src/gromacs/foo.cpp"));
    EXPECT_EQ("src/gromacs/foo.cpp", stripSourcePrefix("some/dir/src/gromacs/foo.cpp"));
    // TODO: For in-source builds, this might not work.
    EXPECT_EQ(std::filesystem::path("src/gromacs/utility/tests/path.cpp").make_preferred(), stripSourcePrefix(__FILE__))
            << "stripSourcePrefix() does not work with compiler-produced file names. "
            << "This only affects source paths reported in fatal error messages.";
}

class PathSearchTest : public testing::TestWithParam<std::string>
{
};

TEST_P(PathSearchTest, SearchOperationsWork)
{
    gmx::test::TestReferenceData    data;
    gmx::test::TestReferenceChecker rootChecker(data.rootChecker());
    std::string                     input = GetParam();

    auto checker = rootChecker.checkCompound("PathToTest", input);
    {
        bool result = false;
        ASSERT_NO_THROW_GMX(result = extensionMatches(input, "pdb"));
        checker.checkBoolean(result, "extensionMatchesPdb");
        // The match is exclusive of the dot separator, so no
        // input string can match.
        ASSERT_FALSE(extensionMatches(input, ".pdb"));
    }
    {
        std::string result;
        ASSERT_NO_THROW_GMX(result = stripExtension(input).generic_u8string());
        checker.checkString(result, "stripExtension");
    }
    {
        std::string result;
        ASSERT_NO_THROW_GMX(result = concatenateBeforeExtension(input, "_34").generic_u8string());
        checker.checkString(result, "concatenateBeforeExtension");
    }
}

INSTANTIATE_TEST_SUITE_P(WithInputPaths,
                         PathSearchTest,
                         testing::Values("",
                                         "md.log",
                                         "md",
                                         "/tmp/absolute.txt",
                                         "simpledir/traj.tng",
                                         "simpledir/traj",
                                         "windowsdir/traj.tng",
                                         "complex.dir/traj.tng",
                                         "complex.dir/traj",
                                         "nested/dir/conf.pdb",
                                         "/tmp/absolutedir/conf.pdb"));

} // namespace
} // namespace test
} // namespace gmx
