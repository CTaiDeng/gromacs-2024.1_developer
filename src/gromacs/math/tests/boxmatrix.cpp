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
 * Tests box matrix inversion routines
 *
 * \todo Test error conditions when they throw exceptions
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_math
 */
#include "gmxpre.h"

#include "gromacs/math/boxmatrix.h"

#include <vector>

#include <gtest/gtest.h>

#include "gromacs/math/vec.h"

#include "testutils/testasserts.h"
namespace gmx
{
namespace test
{
namespace
{


TEST(InvertBoxMatrixTest, IdentityIsImpotent)
{
    matrix in = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

    invertBoxMatrix(in, in);

    EXPECT_REAL_EQ_TOL(in[XX][XX], in[XX][XX], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(in[XX][YY], in[XX][YY], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(in[XX][ZZ], in[XX][ZZ], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(in[YY][XX], in[YY][XX], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(in[YY][YY], in[YY][YY], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(in[YY][ZZ], in[YY][ZZ], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(in[ZZ][XX], in[ZZ][XX], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(in[ZZ][YY], in[ZZ][YY], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(in[ZZ][ZZ], in[ZZ][ZZ], defaultRealTolerance());
}

TEST(InvertBoxMatrixTest, ComputesInverseInPlace)
{
    matrix in       = { { 1, 0, 0 }, { -1, real(2.5), 0 }, { 10, -2, real(1.2) } };
    matrix expected = { { 1, 0, 0 },
                        { real(0.4), real(0.4), 0 },
                        { real(-23.0 / 3.0), real(2.0 / 3.0), real(5.0 / 6.0) } };

    invertBoxMatrix(in, in);

    EXPECT_REAL_EQ_TOL(expected[XX][XX], in[XX][XX], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(expected[XX][YY], in[XX][YY], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(expected[XX][ZZ], in[XX][ZZ], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(expected[YY][XX], in[YY][XX], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(expected[YY][YY], in[YY][YY], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(expected[YY][ZZ], in[YY][ZZ], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(expected[ZZ][XX], in[ZZ][XX], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(expected[ZZ][YY], in[ZZ][YY], defaultRealTolerance());
    EXPECT_REAL_EQ_TOL(expected[ZZ][ZZ], in[ZZ][ZZ], defaultRealTolerance());
}

} // namespace
} // namespace test
} // namespace gmx
