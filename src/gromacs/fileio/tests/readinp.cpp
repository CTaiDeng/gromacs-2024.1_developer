/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * Tests utilities for routines that parse fields e.g. from grompp input
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 */
#include "gmxpre.h"

#include "gromacs/fileio/readinp.h"

#include <gtest/gtest.h>

#include "gromacs/fileio/warninp.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/unique_cptr.h"

namespace gmx
{
namespace testing
{

class ReadTest : public ::testing::Test
{
public:
    ReadTest() :
        inputField_{ { (t_inpfile(0, 0, false, false, false, "test", "")) } }, wi_({ false, 0 })

    {
    }

    std::vector<t_inpfile> inputField_;
    WarningHandler         wi_;
};

TEST_F(ReadTest, get_eint_ReadsInteger)
{
    inputField_.front().value_.assign("1");
    ASSERT_EQ(1, get_eint(&inputField_, "test", 2, &wi_));
    ASSERT_FALSE(warning_errors_exist(wi_));
}

TEST_F(ReadTest, get_eint_WarnsAboutFloat)
{
    inputField_.front().value_.assign("0.8");
    get_eint(&inputField_, "test", 2, &wi_);
    ASSERT_TRUE(warning_errors_exist(wi_));
}

TEST_F(ReadTest, get_eint_WarnsAboutString)
{
    inputField_.front().value_.assign("hello");
    get_eint(&inputField_, "test", 2, &wi_);
    ASSERT_TRUE(warning_errors_exist(wi_));
}

TEST_F(ReadTest, get_eint64_ReadsInteger)
{
    inputField_.front().value_.assign("1");
    ASSERT_EQ(1, get_eint64(&inputField_, "test", 2, &wi_));
    ASSERT_FALSE(warning_errors_exist(wi_));
}

TEST_F(ReadTest, get_eint64_WarnsAboutFloat)
{
    inputField_.front().value_.assign("0.8");
    get_eint64(&inputField_, "test", 2, &wi_);
    ASSERT_TRUE(warning_errors_exist(wi_));
}

TEST_F(ReadTest, get_eint64_WarnsAboutString)
{
    inputField_.front().value_.assign("hello");
    get_eint64(&inputField_, "test", 2, &wi_);
    ASSERT_TRUE(warning_errors_exist(wi_));
}

TEST_F(ReadTest, get_ereal_ReadsInteger)
{
    inputField_.front().value_.assign("1");
    ASSERT_EQ(1, get_ereal(&inputField_, "test", 2, &wi_));
    ASSERT_FALSE(warning_errors_exist(wi_));
}

TEST_F(ReadTest, get_ereal_ReadsFloat)
{
    inputField_.front().value_.assign("0.8");
    ASSERT_EQ(0.8, get_ereal(&inputField_, "test", 2, &wi_));
    ASSERT_FALSE(warning_errors_exist(wi_));
}

TEST_F(ReadTest, get_ereal_WarnsAboutString)
{
    inputField_.front().value_.assign("hello");
    get_ereal(&inputField_, "test", 2, &wi_);
    ASSERT_TRUE(warning_errors_exist(wi_));
}

TEST_F(ReadTest, setStringEntry_ReturnsCorrectString)
{
    const std::string name        = "name";
    const std::string definition  = "definition";
    const std::string returnValue = setStringEntry(&inputField_, name, definition);
    // The definition should be returned
    EXPECT_EQ(returnValue, definition);
    // The name should not be returned
    EXPECT_NE(returnValue, name);
}

} // namespace testing
} // namespace gmx
