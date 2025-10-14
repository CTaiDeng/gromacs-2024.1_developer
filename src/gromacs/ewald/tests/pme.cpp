/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 * Tests for functionality of the pme related classes:
 * class SeparatePmeRanksPermitted
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include <gtest/gtest.h>

#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringcompare.h"

#include "testutils/testasserts.h"
#include "testutils/testmatchers.h"

#include "pmetestcommon.h"

namespace gmx
{

namespace test
{

class SeparatePmeRanksPermittedTest : public ::testing::Test
{
public:
    void disableFirstReason() { separatePmeRanksPermitted_.disablePmeRanks("First reason"); }

    void disableSecondReason() { separatePmeRanksPermitted_.disablePmeRanks("Second reason"); }

    void disableEmptyReason() { separatePmeRanksPermitted_.disablePmeRanks(""); }

protected:
    SeparatePmeRanksPermitted separatePmeRanksPermitted_;
};

TEST_F(SeparatePmeRanksPermittedTest, ZeroPmeDisableReasons)
{
    // Expect that SeparatePmeRanksPermitted is enabled by default
    EXPECT_TRUE(separatePmeRanksPermitted_.permitSeparatePmeRanks());
}

TEST_F(SeparatePmeRanksPermittedTest, CanBeDisabled)
{
    // Test if disablePmeRanks works
    EXPECT_NO_THROW(disableFirstReason(););
}

TEST_F(SeparatePmeRanksPermittedTest, OneDisableReasonFlag)
{
    disableFirstReason();

    // Expect that SeparatePmeRanksPermitted is disabled now
    EXPECT_FALSE(separatePmeRanksPermitted_.permitSeparatePmeRanks());
}

TEST_F(SeparatePmeRanksPermittedTest, OneDisableReasonText)
{
    disableFirstReason();

    // Expect that reasonsWhyDisabled works with one reason
    EXPECT_TRUE(separatePmeRanksPermitted_.reasonsWhyDisabled() == "First reason");
}

TEST_F(SeparatePmeRanksPermittedTest, TwoDisableReasonText)
{
    disableFirstReason();
    disableSecondReason();

    // Expect that reasonsWhyDisabled works with two reasons
    EXPECT_TRUE(separatePmeRanksPermitted_.reasonsWhyDisabled() == "First reason; Second reason");
}

TEST_F(SeparatePmeRanksPermittedTest, EmptyDisableReasonText)
{
    disableEmptyReason();

    // Expect that reasonsWhyDisabled works with empty reason
    EXPECT_TRUE(separatePmeRanksPermitted_.reasonsWhyDisabled().empty());
}

} // namespace test

} // namespace gmx
