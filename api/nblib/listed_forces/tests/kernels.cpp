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
 * This implements kernel tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "listed_forces/kernels.hpp"

#include "testutils/testasserts.h"

namespace nblib
{

TEST(Kernels, HarmonicScalarKernelCanCompute)
{
    real k  = 1.1;
    real x0 = 1.0;
    real x  = 1.2;

    real force, epot;
    std::tie(force, epot) = harmonicScalarForce(k, x0, x);

    EXPECT_REAL_EQ_TOL(-k * (x - x0), force, gmx::test::absoluteTolerance(1e-4));
    EXPECT_REAL_EQ_TOL(0.5 * k * (x - x0) * (x - x0), epot, gmx::test::absoluteTolerance(1e-4));
}

} // namespace nblib
