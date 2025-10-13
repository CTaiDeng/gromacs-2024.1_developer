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
 *
 * \brief Tests routines in strconvert.h.
 *
 * \author Christian Blau <blau@kth.se>
 *
 * \ingroup module_utility
 */

#include "gmxpre.h"

#include "gromacs/utility/strconvert.h"

#include <gtest/gtest.h>

namespace gmx
{
namespace test
{
namespace
{

TEST(StringConvert, NoResultFromEptyString)
{
    const auto parsedArray = parsedArrayFromInputString<float, 3>("");
    EXPECT_FALSE(parsedArray);
}

TEST(StringConvert, ThreeFloatsSuccessfully)
{
    const auto parsedArray = parsedArrayFromInputString<float, 3>("1.2 .5 -6e5");
    EXPECT_FLOAT_EQ((*parsedArray)[0], 1.2);
    EXPECT_FLOAT_EQ((*parsedArray)[1], .5);
    EXPECT_FLOAT_EQ((*parsedArray)[2], -6e5);
}

TEST(StringConvert, OneIntSucessfully)
{
    const auto parsedArray = parsedArrayFromInputString<int, 1>(" 1 \t  ");
    EXPECT_FLOAT_EQ((*parsedArray)[0], 1);
}

TEST(StringConvert, FloatAsStringToIntArrayThrows)
{
    const auto& toTest = []() { return parsedArrayFromInputString<int, 1>(" 1.2 "); };
    EXPECT_THROW(toTest(), InvalidInputError);
}

TEST(StringConvert, ThrowsWhenWrongSize)
{
    // use the lambda due to aviod Macro substitution error with template function
    const auto& toTest = []() { return parsedArrayFromInputString<float, 2>("1.2\t\n  .5 -6e5"); };
    EXPECT_THROW(toTest(), InvalidInputError);
}

TEST(StringConvert, StringIdentityTransformWithArrayThrows)
{
    // use the lambda due to aviod Macro substitution error with template function
    const auto& toTest = []() {
        return stringIdentityTransformWithArrayCheck<float, 3>(
                "-10 5 4 1", "Here, I explain where the error occurred: ");
    };
    EXPECT_THROW(toTest(), InvalidInputError);
}

TEST(StringConvert, StringIdentityTransformWithArrayOkay)
{
    // use the lambda due to aviod Macro substitution error with template function
    const std::string input("1.2\t\n  .5 -6e5");
    const std::string output = stringIdentityTransformWithArrayCheck<float, 3>(
            input, "Here, I explain where the error occurred: ");
    EXPECT_EQ(input, output);
}


} // namespace
} // namespace test
} // namespace gmx
