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
 * This PbcHolder tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include <cmath>

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

#include "nblib/pbc.hpp"

using gmx::test::defaultRealTolerance;

namespace nblib
{

TEST(NBlibTest, PbcHolderWorks)
{
    Box box(10, 10, 10);

    PbcHolder pbcHolder(PbcType::Xyz, box);

    gmx::RVec x1{ 1.0, 1.1, 0.9 }, x2{ 9, 8.9, 9.1 };
    gmx::RVec dx;

    pbcHolder.dxAiuc(x1, x2, dx);
    gmx::RVec ref{ 2, 2.2, 1.8 };

    EXPECT_REAL_EQ_TOL(ref[0], dx[0], gmx::test::relativeToleranceAsFloatingPoint(ref[0], 1e-6));
    EXPECT_REAL_EQ_TOL(ref[1], dx[1], gmx::test::relativeToleranceAsFloatingPoint(ref[0], 1e-6));
    EXPECT_REAL_EQ_TOL(ref[2], dx[2], gmx::test::relativeToleranceAsFloatingPoint(ref[0], 1e-6));
}

} // namespace nblib
