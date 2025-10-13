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
 * This implements basic nblib utility tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */

#ifndef NBLIB_LINEAR_CHAIN_DATA_HPP
#define NBLIB_LINEAR_CHAIN_DATA_HPP

#include "listed_forces/traits.h"

#include "nblib/box.h"

#include "topologyhelpers.h"

namespace nblib
{

//! \brief sets up an interaction tuples for a linear chain with nParticles
class LinearChainData
{
public:
    LinearChainData(int nP, float outlierRatio = 0.02) : nParticles(nP)
    {
        HarmonicBondType              bond1{ 376560, real(0.1001 * std::sqrt(3)) };
        HarmonicBondType              bond2{ 313800, real(0.1001 * std::sqrt(3)) };
        std::vector<HarmonicBondType> bonds{ bond1, bond2 };
        pickType<HarmonicBondType>(interactions).parameters = bonds;

        HarmonicAngle              angle(397.5, Degrees(179.9));
        std::vector<HarmonicAngle> angles{ angle };
        pickType<HarmonicAngle>(interactions).parameters = angles;

        std::vector<InteractionIndex<HarmonicBondType>> bondIndices;
        for (int i = 0; i < nParticles - 1; ++i)
        {
            // third index: alternate between the two bond parameters
            bondIndices.push_back(InteractionIndex<HarmonicBondType>{ i, i + 1, i % 2 });
        }
        pickType<HarmonicBondType>(interactions).indices = bondIndices;

        std::vector<InteractionIndex<HarmonicAngle>> angleIndices;
        for (int i = 0; i < nParticles - 2; ++i)
        {
            angleIndices.push_back(InteractionIndex<HarmonicAngle>{ i, i + 1, i + 2, 0 });
        }
        pickType<HarmonicAngle>(interactions).indices = angleIndices;

        // initialize coordinates
        x.resize(nParticles);
        for (int i = 0; i < nParticles; ++i)
        {
            x[i] = real(i) * gmx::RVec{ 0.1, 0.1, 0.1 };
        }

        forces = std::vector<gmx::RVec>(nParticles, gmx::RVec{ 0, 0, 0 });

        box.reset(new Box(0.1 * (nParticles + 1), 0.1 * (nParticles + 1), 0.1 * (nParticles + 1)));

        addOutliers(outlierRatio);
    }

    // void createAggregates() { aggregateTransformations(interactions); }

    void addOutliers(float outlierRatio)
    {
        HarmonicBondType dummyBond{ 1e-6, real(0.1001 * std::sqrt(3)) };
        pickType<HarmonicBondType>(interactions).parameters.push_back(dummyBond);
        int bondIndex = pickType<HarmonicBondType>(interactions).parameters.size() - 1;

        int nOutliers = nParticles * outlierRatio;
        srand(42);
        for (int s = 0; s < nOutliers; ++s)
        {
            int from = rand() % nParticles;
            int to   = rand() % nParticles;
            if (from != to)
            {
                pickType<HarmonicBondType>(interactions)
                        .indices.push_back(InteractionIndex<HarmonicBondType>{ from, to, bondIndex });
            }
        }
    }

    int nParticles;

    std::vector<gmx::RVec> x;
    std::vector<gmx::RVec> forces;

    ListedInteractionData interactions;

    std::shared_ptr<Box> box;
};

} // namespace nblib

#endif // NBLIB_LINEAR_CHAIN_DATA_HPP
