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

#include "gmxpre.h"

#include "gromacs/utility/template_mp.h"

#include <gtest/gtest.h>

namespace gmx
{
namespace
{

enum class Options
{
    Op0   = 0,
    Op1   = 1,
    Op2   = 2,
    Count = 3
};

template<Options i, Options j>
static int testEnumTwoIPlusJPlusK(int k)
{
    return 2 * int(i) + int(j) + k;
}

template<bool doDoubling, Options i, Options j>
static int testBoolEnumTwoIPlusJPlusK(int k)
{
    return (doDoubling ? 2 : 1) * int(i) + int(j) + k;
}

template<bool doDoubling>
static int testBoolDoubleOrNot(int k)
{
    return (doDoubling ? 2 : 1) * k;
}


TEST(TemplateMPTest, DispatchTemplatedFunctionEnum)
{
    int five           = 5;
    int two1plus2plus5 = dispatchTemplatedFunction(
            [=](auto p1, auto p2) { return testEnumTwoIPlusJPlusK<p1, p2>(five); }, Options::Op1, Options::Op2);
    EXPECT_EQ(two1plus2plus5, 9);
}

TEST(TemplateMPTest, DispatchTemplatedFunctionBool)
{
    int five = 5;
    int double5 = dispatchTemplatedFunction([=](auto p1) { return testBoolDoubleOrNot<p1>(five); }, true);
    EXPECT_EQ(double5, 10);
    int just5 = dispatchTemplatedFunction([=](auto p1) { return testBoolDoubleOrNot<p1>(five); }, false);
    EXPECT_EQ(just5, 5);
}

TEST(TemplateMPTest, DispatchTemplatedFunctionEnumBool)
{
    int five           = 5;
    int two1plus2plus5 = dispatchTemplatedFunction(
            [=](auto p1, auto p2, auto p3) { return testBoolEnumTwoIPlusJPlusK<p1, p2, p3>(five); },
            true,
            Options::Op1,
            Options::Op2);
    EXPECT_EQ(two1plus2plus5, 9);
}

} // anonymous namespace
} // namespace gmx
