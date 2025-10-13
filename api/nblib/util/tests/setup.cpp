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
 * \brief
 * This implements basic nblib utility tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "nblib/util/setup.h"

#include <vector>

#include "testutils/testasserts.h"

#include "testhelpers.h"


namespace nblib
{
namespace test
{
namespace
{

TEST(NBlibTest, isRealValued)
{
    std::vector<Vec3> vec;
    vec.emplace_back(1., 1., 1.);
    vec.emplace_back(2., 2., 2.);

    bool ret = isRealValued(vec);
    EXPECT_EQ(ret, true);
}

TEST(NBlibTest, checkNumericValuesHasNan)
{
    std::vector<Vec3> vec;
    vec.emplace_back(1., 1., 1.);
    vec.emplace_back(2., 2., 2.);

    vec.emplace_back(NAN, NAN, NAN);

    bool ret = isRealValued(vec);
    EXPECT_EQ(ret, false);
}

TEST(NBlibTest, checkNumericValuesHasInf)
{
    std::vector<Vec3> vec;
    vec.emplace_back(1., 1., 1.);
    vec.emplace_back(2., 2., 2.);

    vec.emplace_back(INFINITY, INFINITY, INFINITY);

    bool ret = isRealValued(vec);
    EXPECT_EQ(ret, false);
}


TEST(NBlibTest, GeneratedVelocitiesAreCorrect)
{
    constexpr size_t  N = 10;
    std::vector<real> masses(N, 1.0);
    std::vector<Vec3> velocities;
    velocities = generateVelocity(300.0, 1, masses);

    RefDataChecker velocitiesTest;
    velocitiesTest.testArrays<Vec3>(velocities, "generated-velocities");
}
TEST(NBlibTest, generateVelocitySize)
{
    constexpr int     N = 10;
    std::vector<real> masses(N, 1.0);
    auto              out = generateVelocity(300.0, 1, masses);
    EXPECT_EQ(out.size(), N);
}

TEST(NBlibTest, generateVelocityCheckNumbers)
{
    constexpr int     N = 10;
    std::vector<real> masses(N, 1.0);
    auto              out = generateVelocity(300.0, 1, masses);
    bool              ret = isRealValued(out);
    EXPECT_EQ(ret, true);
}

} // namespace
} // namespace test
} // namespace nblib
