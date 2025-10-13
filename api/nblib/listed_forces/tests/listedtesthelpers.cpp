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
 * A collection of helper utilities that allow setting up both Nblib and
 * GROMACS fixtures for computing listed interactions given sets of parameters
 * and coordinates
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */

#include "listedtesthelpers.h"

#include "gromacs/utility/arrayref.h"

#include "testutils/testasserts.h"
#include "testutils/testmatchers.h"

#include "gmxcalculator.h"

namespace nblib
{

void compareNblibAndGmxListedImplementations(const ListedInteractionData&  interactionData,
                                             const std::vector<gmx::RVec>& coordinates,
                                             size_t                        numParticles,
                                             int                           numThreads,
                                             const Box&                    box,
                                             real                          tolerance)
{
    ListedForceCalculator calculator(interactionData, numParticles, numThreads, box);

    std::vector<gmx::RVec>            forces(numParticles, gmx::RVec{ 0, 0, 0 });
    std::vector<gmx::RVec>            shiftForces(gmx::c_numShiftVectors, gmx::RVec{ 0, 0, 0 });
    ListedForceCalculator::EnergyType energies;

    calculator.compute(coordinates, forces, shiftForces, energies, true);

    ListedGmxCalculator gmxCalculator(interactionData, numParticles, numThreads, box);

    std::vector<gmx::RVec>            gmxForces(numParticles, gmx::RVec{ 0, 0, 0 });
    std::vector<gmx::RVec>            gmxShiftForces(gmx::c_numShiftVectors, gmx::RVec{ 0, 0, 0 });
    ListedForceCalculator::EnergyType gmxEnergies;

    gmxCalculator.compute(coordinates, gmxForces, gmxShiftForces, gmxEnergies, true);

    gmx::test::FloatingPointTolerance tolSetting(tolerance, tolerance, 1.0e-5, 1.0e-8, 200, 100, false);

    EXPECT_THAT(forces, Pointwise(gmx::test::RVecEq(tolSetting), gmxForces));
    EXPECT_THAT(shiftForces, Pointwise(gmx::test::RVecEq(tolSetting), gmxShiftForces));
}

} // namespace nblib
