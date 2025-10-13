/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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
 * Tests structure similarity measures rmsd and size-independent rho factor.
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \ingroup module_math
 */
#include "gmxpre.h"

#include <array>

#include <gtest/gtest.h>

#include "gromacs/math/do_fit.h"
#include "gromacs/math/vec.h"

#include "testutils/testasserts.h"

namespace
{

using gmx::RVec;
using gmx::test::defaultRealTolerance;
class StructureSimilarityTest : public ::testing::Test
{
protected:
    static constexpr int       c_nAtoms = 4;
    std::array<RVec, c_nAtoms> structureA_{ { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0, 0 } } };
    std::array<RVec, c_nAtoms> structureB_{ { { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 0, 0 } } };
    std::array<real, c_nAtoms> masses_{ { 1, 1, 1, 0 } };
    std::array<int, 3>         index_{ { 0, 1, 2 } };
    rvec*                      x1_ = gmx::as_rvec_array(structureA_.data());
    rvec*                      x2_ = gmx::as_rvec_array(structureB_.data());
    real*                      m_  = masses_.data();
};

TEST_F(StructureSimilarityTest, StructureComparedToSelfHasZeroRMSD)
{
    EXPECT_REAL_EQ_TOL(0., rmsdev(c_nAtoms, m_, x1_, x1_), defaultRealTolerance());
}

TEST_F(StructureSimilarityTest, StructureComparedToSelfHasZeroRho)
{
    EXPECT_REAL_EQ_TOL(0., rhodev(c_nAtoms, m_, x1_, x1_), defaultRealTolerance());
}

TEST_F(StructureSimilarityTest, YieldsCorrectRMSD)
{
    EXPECT_REAL_EQ_TOL(sqrt(2.0), rmsdev(c_nAtoms, m_, x1_, x2_), defaultRealTolerance());
}

TEST_F(StructureSimilarityTest, YieldsCorrectRho)
{
    EXPECT_REAL_EQ_TOL(2., rhodev(c_nAtoms, m_, x1_, x2_), defaultRealTolerance());
}

TEST_F(StructureSimilarityTest, YieldsCorrectRMSDWithIndex)
{
    EXPECT_REAL_EQ_TOL(
            sqrt(2.0), rmsdev_ind(index_.size(), index_.data(), m_, x1_, x2_), defaultRealTolerance());
}

TEST_F(StructureSimilarityTest, YieldsCorrectRhoWidthIndex)
{
    EXPECT_REAL_EQ_TOL(2., rhodev_ind(index_.size(), index_.data(), m_, x1_, x2_), defaultRealTolerance());
}

} // namespace
