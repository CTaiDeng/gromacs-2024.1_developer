/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * \brief Tests for GROMACS uniform real distribution
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \ingroup module_random
 */
#include "gmxpre.h"

#include "gromacs/random/uniformrealdistribution.h"

#include <gtest/gtest.h>

#include "gromacs/random/threefry.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace gmx
{

namespace
{

TEST(UniformRealDistributionTest, GenerateCanonical)
{
    gmx::test::TestReferenceData    data;
    gmx::test::TestReferenceChecker checker(data.rootChecker());

    gmx::ThreeFry2x64<8> rng(123456, gmx::RandomDomain::Other);
    std::vector<real>    result;

    result.reserve(10);
    for (int i = 0; i < 10; i++)
    {
        result.push_back(gmx::generateCanonical<real, std::numeric_limits<real>::digits>(rng));
    }
    checker.checkSequence(result.begin(), result.end(), "GenerateCanonical");
}

TEST(UniformRealDistributionTest, Output)
{
    gmx::test::TestReferenceData    data;
    gmx::test::TestReferenceChecker checker(data.rootChecker());

    gmx::ThreeFry2x64<8>               rng(123456, gmx::RandomDomain::Other);
    gmx::UniformRealDistribution<real> dist(1.0, 10.0);
    std::vector<real>                  result;

    result.reserve(10);
    for (int i = 0; i < 10; i++)
    {
        result.push_back(dist(rng));
    }
    checker.checkSequence(result.begin(), result.end(), "UniformRealDistribution");
}

TEST(UniformRealDistributionTest, Logical)
{
    gmx::ThreeFry2x64<8>               rng(123456, gmx::RandomDomain::Other);
    gmx::UniformRealDistribution<real> distA(2.0, 5.0);
    gmx::UniformRealDistribution<real> distB(2.0, 5.0);
    gmx::UniformRealDistribution<real> distC(3.0, 5.0);
    gmx::UniformRealDistribution<real> distD(2.0, 4.0);

    EXPECT_EQ(distA, distB);
    EXPECT_NE(distA, distC);
    EXPECT_NE(distA, distD);
}


TEST(UniformRealDistributionTest, Reset)
{
    gmx::ThreeFry2x64<8>                            rng(123456, gmx::RandomDomain::Other);
    gmx::UniformRealDistribution<real>              distA(2.0, 5.0);
    gmx::UniformRealDistribution<real>              distB(2.0, 5.0);
    gmx::UniformRealDistribution<real>::result_type valA, valB;

    valA = distA(rng);

    distB(rng);
    rng.restart();
    distB.reset();

    valB = distB(rng);

    // Use floating-point test with no tolerance rather than exact
    // test, since the later fails with 32-bit gcc-4.8.4
    EXPECT_REAL_EQ_TOL(valA, valB, gmx::test::ulpTolerance(0));
}

TEST(UniformRealDistributionTest, AltParam)
{
    gmx::ThreeFry2x64<8>                            rngA(123456, gmx::RandomDomain::Other);
    gmx::ThreeFry2x64<8>                            rngB(123456, gmx::RandomDomain::Other);
    gmx::UniformRealDistribution<real>              distA(2.0, 5.0);
    gmx::UniformRealDistribution<real>              distB; // default parameters
    gmx::UniformRealDistribution<real>::param_type  paramA(2.0, 5.0);
    gmx::UniformRealDistribution<real>::result_type valA, valB;

    EXPECT_NE(distA(rngA), distB(rngB));
    rngA.restart();
    rngB.restart();
    distA.reset();
    distB.reset();

    valA = distA(rngA);
    valB = distB(rngB, paramA);

    // Use floating-point test with no tolerance rather than exact test,
    // since the latter fails with 32-bit gcc-4.8.4
    EXPECT_REAL_EQ_TOL(valA, valB, gmx::test::ulpTolerance(0));
}

} // namespace

} // namespace gmx
