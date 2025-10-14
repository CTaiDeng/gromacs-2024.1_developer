/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Tests utilities for testing xvg files
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include "testutils/xvgtest.h"

#include <vector>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/stringstream.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

//! Input testing data - an inline xvg file.
const char* const input[] = { "0     2905.86    -410.199",   "0.2     6656.67    -430.437",
                              "0.4     5262.44    -409.399", "0.6     5994.69    -405.763",
                              "0.8     5941.37    -408.337", "1     5869.87    -411.124" };

TEST(XvgTests, CreateFile)
{
    {
        // Create new data
        TestReferenceData    data(ReferenceDataMode::UpdateAll);
        TestReferenceChecker checker(data.rootChecker());
        // Convert char array to a stream and add it to the checker
        gmx::StringInputStream sis(input);
        checkXvgFile(&sis, &checker, XvgMatchSettings());
    }
    {
        // Now read it back
        TestReferenceData    data(ReferenceDataMode::Compare);
        TestReferenceChecker checker(data.rootChecker());
        // Convert char array to a stream and add it to the checker
        gmx::StringInputStream sis(input);
        checkXvgFile(&sis, &checker, XvgMatchSettings());
    }
}

TEST(XvgTests, CheckMissing)
{
    {
        // Create new data
        TestReferenceData    data(ReferenceDataMode::UpdateAll);
        TestReferenceChecker checker(data.rootChecker());
        // Convert char array to a stream and add it to the checker
        gmx::StringInputStream sis(input);
        checkXvgFile(&sis, &checker, XvgMatchSettings());
    }
    {
        const char* const input[] = { "0     2905.86    -410.199",
                                      "0.2     6656.67    -430.437",
                                      "0.4     5262.44    -409.399" };
        // Now check with missing data
        TestReferenceData      data(ReferenceDataMode::Compare);
        TestReferenceChecker   checker(data.rootChecker());
        gmx::StringInputStream sis(input);
        EXPECT_NONFATAL_FAILURE(checkXvgFile(&sis, &checker, XvgMatchSettings()),
                                "not used in test");
    }
}

TEST(XvgTests, CheckExtra)
{
    {
        // Create new data
        TestReferenceData    data(ReferenceDataMode::UpdateAll);
        TestReferenceChecker checker(data.rootChecker());
        // Convert char array to a stream and add it to the checker
        gmx::StringInputStream sis(input);
        checkXvgFile(&sis, &checker, XvgMatchSettings());
    }
    {
        const char* const input[] = { "0     2905.86    -410.199",   "0.2     6656.67    -430.437",
                                      "0.4     5262.44    -409.399", "0.6     5994.69    -405.763",
                                      "0.8     5941.37    -408.337", "1     5869.87    -411.124",
                                      "1.2     5889.87    -413.124" };
        // Now check with missing data
        TestReferenceData      data(ReferenceDataMode::Compare);
        TestReferenceChecker   checker(data.rootChecker());
        gmx::StringInputStream sis(input);
        EXPECT_NONFATAL_FAILURE(checkXvgFile(&sis, &checker, XvgMatchSettings()), "Row6");
    }
}

TEST(XvgTests, ReadIncorrect)
{
    {
        // Create new data
        TestReferenceData    data(ReferenceDataMode::UpdateAll);
        TestReferenceChecker checker(data.rootChecker());
        // Convert char array to a stream and add it to the checker
        gmx::StringInputStream sis(input);
        checkXvgFile(&sis, &checker, XvgMatchSettings());
    }
    {
        const char* const input[] = { "0     2905.86    -410.199",   "0.2     6656.67    -430.437",
                                      "0.4     5262.44    -409.399", "0.6     5994.69    -405.763",
                                      "0.8     5941.37    -408.337", "1     5869.87    -421.124" };
        // Now check with incorrect data
        TestReferenceData      data(ReferenceDataMode::Compare);
        TestReferenceChecker   checker(data.rootChecker());
        gmx::StringInputStream sis(input);
        EXPECT_NONFATAL_FAILURE(checkXvgFile(&sis, &checker, XvgMatchSettings()), "-411");
    }
}

} // namespace
} // namespace test
} // namespace gmx
