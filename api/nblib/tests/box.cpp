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
 * This implements basic nblib box tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "nblib/box.h"

#include <cmath>

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

#include "nblib/exception.h"

using gmx::test::defaultRealTolerance;

namespace nblib
{

TEST(NBlibTest, CubicBoxCannotHaveNaN)
{
    real number = NAN;
    EXPECT_THROW(Box box(number), InputException);
}

TEST(NBlibTest, CubicBoxCannotHaveInf)
{
    real number = INFINITY;
    EXPECT_THROW(Box box(number), InputException);
}

TEST(NBlibTest, RectangularBoxCannotHaveNaN)
{
    real number = NAN;
    EXPECT_THROW(Box box(number, real(1.), real(1.)), InputException);
}

TEST(NBlibTest, RectangularBoxCannotHaveInf)
{
    real number = INFINITY;
    EXPECT_THROW(Box box(number, real(1.), real(1.)), InputException);
}

TEST(NBlibTest, CubicBoxWorks)
{
    real              length = 3;
    Box::LegacyMatrix ref    = { { length, 0, 0 }, { 0, length, 0 }, { 0, 0, length } };
    Box               test   = Box(length);

    for (int i = 0; i < dimSize; ++i)
    {
        for (int j = 0; j < dimSize; ++j)
        {
            EXPECT_REAL_EQ_TOL(ref[i][j], test.legacyMatrix()[i][j], defaultRealTolerance());
        }
    }
}

TEST(NBlibTest, BoxEqual)
{
    {
        Box a(0), b(0);
        EXPECT_TRUE(a == b);
    }
    {
        Box a(1), b(1);
        EXPECT_TRUE(a == b);
    }
    {
        Box a(1, 2, 3), b(1, 2, 3);
        EXPECT_TRUE(a == b);
    }
    {
        Box a(0, 2, 3), b(1, 2, 3);
        EXPECT_FALSE(a == b);
    }
}

} // namespace nblib
