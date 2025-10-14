/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 * Tests for the Range class.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/range.h"

#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "testutils/testasserts.h"

namespace gmx
{

namespace
{

TEST(Range, EmptyRangeWorks)
{
    Range<int> range;

    EXPECT_EQ(range.empty(), true);
    EXPECT_EQ(range.size(), 0);
}

TEST(Range, NonEmptyRangeWorks)
{
    const Range<char> range(3, 5);

    EXPECT_EQ(range.empty(), false);
    EXPECT_EQ(range.size(), 2);
}

TEST(Range, BeginEnd)
{
    const Range<long> range(-2, 9);

    EXPECT_EQ(range.begin(), -2);
    EXPECT_EQ(*range.begin(), -2);
    EXPECT_EQ(range.end(), 9);
    EXPECT_EQ(*range.end(), 9);
}

TEST(Range, IsInRangeWorks)
{
    const Range<size_t> range(5, 8);

    EXPECT_EQ(range.isInRange(4), false);
    EXPECT_EQ(range.isInRange(5), true);
    EXPECT_EQ(range.isInRange(6), true);
    EXPECT_EQ(range.isInRange(7), true);
    EXPECT_EQ(range.isInRange(8), false);
}

TEST(Range, IteratorWorks)
{
    const Range<Index> range(-1, 3);

    int minValue = std::numeric_limits<int>::max();
    int maxValue = std::numeric_limits<int>::min();
    for (int i : range)
    {
        minValue = std::min(minValue, i);
        maxValue = std::max(maxValue, i);
    }
    EXPECT_EQ(minValue, -1);
    EXPECT_EQ(maxValue, 2);
}

} // namespace

} // namespace gmx
