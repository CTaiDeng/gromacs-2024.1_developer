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
 * This implements basic nblib box tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */

#include "testutils/testasserts.h"

#include "listedtesthelpers.h"
#include "pbc.hpp"

namespace nblib
{

static ListedInteractionData createInteractionData(int numCoordinates)
{
    ListedInteractionData interactionData;

    HarmonicBondType harmonicBond(2.0, 1.0);
    HarmonicAngle    harmonicAngle(2.2, Degrees(91.0));
    ProperDihedral   properDihedral(Degrees(45), 2.3, 1);

    for (int i = 0; i < numCoordinates; ++i)
    {
        for (int j = i + 1; j < numCoordinates; ++j)
        {
            pickType<HarmonicBondType>(interactionData).indices.push_back({ i, j, 0 });
        }
    }
    pickType<HarmonicBondType>(interactionData).parameters.push_back(harmonicBond);

    for (int i = 0; i < numCoordinates; ++i)
    {
        for (int j = i + 1; j < numCoordinates; ++j)
        {
            for (int k = j + 1; k < numCoordinates; ++k)
            {
                pickType<HarmonicAngle>(interactionData).indices.push_back({ i, j, k, 0 });
            }
        }
    }
    pickType<HarmonicAngle>(interactionData).parameters.push_back(harmonicAngle);

    for (int i = 0; i < numCoordinates; ++i)
    {
        for (int j = i + 1; j < numCoordinates; ++j)
        {
            for (int k = j + 1; k < numCoordinates; ++k)
            {
                for (int l = k + 1; l < numCoordinates; ++l)
                {
                    pickType<ProperDihedral>(interactionData).indices.push_back({ i, j, k, l, 0 });
                }
            }
        }
    }
    pickType<ProperDihedral>(interactionData).parameters.push_back(properDihedral);

    return interactionData;
}

static std::vector<gmx::RVec> createTestCoordinates(int numParticles)
{
    std::vector<gmx::RVec> coordinates(numParticles);
    for (auto& c : coordinates)
    {
        c[0] = drand48();
        c[1] = drand48();
        c[2] = drand48();
    }

    return coordinates;
}

TEST(NBlibTest, shiftForcesAreCorrect)
{
    int                   numParticles    = 30;
    ListedInteractionData interactionData = createInteractionData(numParticles);

    Box  box(1.0);
    auto coordinates = createTestCoordinates(numParticles);

    compareNblibAndGmxListedImplementations(interactionData, coordinates, numParticles, 1, box, 1e-3);
}

} // namespace nblib
