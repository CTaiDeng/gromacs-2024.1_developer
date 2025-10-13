/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 * This implements virials tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "nblib/virials.h"

#include <gtest/gtest.h>

#include "gromacs/mdtypes/forcerec.h"

#include "nblib/box.h"
#include "nblib/nbnxmsetuphelpers.h"

namespace nblib
{
namespace test
{

TEST(VirialsTest, computeVirialTensorWorks)
{
    std::vector<Vec3> coords = { { 0, 1, 2 }, { 2, 3, 4 } };
    std::vector<Vec3> forces = { { 2, 1, 2 }, { 4, 3, 4 } };
    std::vector<Vec3> shiftForces(gmx::c_numShiftVectors, Vec3(0.0, 1.0, 0.0));
    Box               box(1, 2, 3);
    t_forcerec        forcerec;
    updateForcerec(&forcerec, box.legacyMatrix());
    std::vector<Vec3> shiftVectors(gmx::c_numShiftVectors);
    // copy shift vectors from ForceRec
    std::copy(forcerec.shift_vec.begin(), forcerec.shift_vec.end(), shiftVectors.begin());
    std::vector<real> virialTest(9, 0);
    computeVirialTensor(coords, forces, shiftVectors, shiftForces, box, virialTest);
    std::vector<real> virialRef{ -4, -3, -4, -7, -5, -7, -10, -7, -10 };
    EXPECT_EQ(virialRef, virialTest);
}

} // namespace test

} // namespace nblib
