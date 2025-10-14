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
 * This implements basic nblib utility tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "listed_forces/transformations.h"

#include <numeric>

#include <gtest/gtest.h>

#include "listed_forces/traits.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace nblib
{
namespace test
{
namespace
{

ListedInteractionData unsortedInteractions()
{
    ListedInteractionData interactions;

    std::vector<InteractionIndex<HarmonicBondType>> bondIndices{ { 0, 2, 0 }, { 0, 1, 0 } };
    pickType<HarmonicBondType>(interactions).indices = std::move(bondIndices);

    std::vector<InteractionIndex<HarmonicAngle>> angleIndices{ { 0, 1, 2, 0 }, { 1, 0, 2, 0 } };
    pickType<HarmonicAngle>(interactions).indices = std::move(angleIndices);

    std::vector<InteractionIndex<ProperDihedral>> dihedralIndices{ { 0, 2, 1, 3, 0 }, { 0, 1, 2, 3, 0 } };
    pickType<ProperDihedral>(interactions).indices = std::move(dihedralIndices);

    return interactions;
}

TEST(ListedTransformations, SortInteractionIndices)
{
    ListedInteractionData interactions = unsortedInteractions();
    sortInteractions(interactions);

    std::vector<InteractionIndex<HarmonicBondType>> refBondIndices{ { 0, 1, 0 }, { 0, 2, 0 } };
    std::vector<InteractionIndex<HarmonicAngle>>  refAngleIndices{ { 1, 0, 2, 0 }, { 0, 1, 2, 0 } };
    std::vector<InteractionIndex<ProperDihedral>> refDihedralIndices{ { 0, 1, 2, 3, 0 },
                                                                      { 0, 2, 1, 3, 0 } };

    EXPECT_EQ(pickType<HarmonicBondType>(interactions).indices, refBondIndices);
    EXPECT_EQ(pickType<HarmonicAngle>(interactions).indices, refAngleIndices);
    EXPECT_EQ(pickType<ProperDihedral>(interactions).indices, refDihedralIndices);
}

} // namespace
} // namespace test
} // namespace nblib
