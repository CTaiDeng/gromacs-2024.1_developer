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
 * Tests for time control value setting.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_fileio
 */
#include "gmxpre.h"

#include "gromacs/fileio/timecontrol.h"

#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "testutils/include/testutils/testasserts.h"

TEST(TimeControlTest, UnSetHasNoValue)
{
    auto value = timeValue(TimeControl::Begin);
    EXPECT_FALSE(value.has_value());
}

TEST(TimeControlTest, CanSetValue)
{
    setTimeValue(TimeControl::Begin, 13.37);
    auto value = timeValue(TimeControl::Begin);
    ASSERT_TRUE(value.has_value());
    EXPECT_FLOAT_EQ(*value, 13.37);
    auto otherValue = timeValue(TimeControl::End);
    EXPECT_FALSE(otherValue.has_value());
}

TEST(TimeControlTest, CanUnsetValueAgain)
{
    setTimeValue(TimeControl::Begin, 13.37);
    setTimeValue(TimeControl::End, 42.23);
    auto value      = timeValue(TimeControl::Begin);
    auto otherValue = timeValue(TimeControl::End);
    EXPECT_TRUE(value.has_value());
    EXPECT_TRUE(otherValue.has_value());
    unsetTimeValue(TimeControl::Begin);
    auto newValue      = timeValue(TimeControl::Begin);
    auto newOtherValue = timeValue(TimeControl::End);
    EXPECT_FALSE(newValue.has_value());
    EXPECT_TRUE(newOtherValue.has_value());
}
