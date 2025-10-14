/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * Tests for the exponential moving average.
 *
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_math
 */
#include "gmxpre.h"

#include "gromacs/math/exponentialmovingaverage.h"

#include <gtest/gtest.h>

#include "testutils/testasserts.h"

namespace gmx
{

namespace test
{

namespace
{

TEST(ExponentialMovingAverage, ThrowsWhenLagTimeIsZero)
{
    EXPECT_THROW_GMX(ExponentialMovingAverage(0), InconsistentInputError);
}

TEST(ExponentialMovingAverage, ThrowsWhenLagTimeIsNegative)
{
    EXPECT_THROW_GMX(ExponentialMovingAverage(-10), InconsistentInputError);
}

TEST(ExponentialMovingAverage, LagTimeOneYieldsInstantaneousValue)
{
    const real               lagTime = 1;
    ExponentialMovingAverage exponentialMovingAverage(lagTime);

    exponentialMovingAverage.updateWithDataPoint(10);
    EXPECT_REAL_EQ(10, exponentialMovingAverage.biasCorrectedAverage());

    exponentialMovingAverage.updateWithDataPoint(-10);
    EXPECT_REAL_EQ(-10, exponentialMovingAverage.biasCorrectedAverage());
}

TEST(ExponentialMovingAverage, YieldsCorrectValue)
{
    const real               lagTime = 100;
    ExponentialMovingAverage exponentialMovingAverage(lagTime);

    exponentialMovingAverage.updateWithDataPoint(10);
    EXPECT_REAL_EQ(10, exponentialMovingAverage.biasCorrectedAverage());

    exponentialMovingAverage.updateWithDataPoint(-10);
    EXPECT_REAL_EQ(-0.050251256281406857, exponentialMovingAverage.biasCorrectedAverage());

    exponentialMovingAverage.updateWithDataPoint(0);
    EXPECT_REAL_EQ(-0.03333221103666531, exponentialMovingAverage.biasCorrectedAverage());
}

TEST(ExponentialMovingAverage, SetAverageCorrectly)
{
    const real               lagTime = 100;
    ExponentialMovingAverage exponentialMovingAverage(lagTime);

    exponentialMovingAverage.updateWithDataPoint(10);
    EXPECT_REAL_EQ(10, exponentialMovingAverage.biasCorrectedAverage());

    ExponentialMovingAverageState thisState = exponentialMovingAverage.state();

    ExponentialMovingAverage other(lagTime, thisState);

    other.updateWithDataPoint(-10);
    EXPECT_REAL_EQ(-0.050251256281406857, other.biasCorrectedAverage());

    other.updateWithDataPoint(0);
    EXPECT_REAL_EQ(-0.03333221103666531, other.biasCorrectedAverage());
}

TEST(ExponentialMovingAverage, DeterminesCorrectlyIfIncreasing)
{
    const real               lagTime = 100;
    ExponentialMovingAverage exponentialMovingAverage(lagTime);

    exponentialMovingAverage.updateWithDataPoint(10);
    exponentialMovingAverage.updateWithDataPoint(9.99);

    EXPECT_FALSE(exponentialMovingAverage.increasing());

    exponentialMovingAverage.updateWithDataPoint(-10);

    EXPECT_FALSE(exponentialMovingAverage.increasing());

    exponentialMovingAverage.updateWithDataPoint(100);
    EXPECT_TRUE(exponentialMovingAverage.increasing());
}


TEST(ExponentialMovingAverage, InverseLagTimeCorrect)
{
    const real               lagTime = 2.;
    ExponentialMovingAverage exponentialMovingAverage(lagTime);
    EXPECT_REAL_EQ(0.5, exponentialMovingAverage.inverseTimeConstant());
}

TEST(ExponentialMovingAverage, RoundTripAsKeyValueTree)
{
    KeyValueTreeBuilder           builder;
    const real                    weightedSum   = 9;
    const real                    weightedCount = 1;
    const bool                    increasing    = true;
    ExponentialMovingAverageState state         = { weightedSum, weightedCount, increasing };
    exponentialMovingAverageStateAsKeyValueTree(builder.rootObject(), state);
    state                     = {};
    KeyValueTreeObject result = builder.build();
    state                     = exponentialMovingAverageStateFromKeyValueTree(result);
    EXPECT_EQ(weightedSum, state.weightedSum_);
    EXPECT_EQ(weightedCount, state.weightedCount_);
    EXPECT_EQ(increasing, state.increasing_);
}

} // namespace

} // namespace test

} // namespace gmx
