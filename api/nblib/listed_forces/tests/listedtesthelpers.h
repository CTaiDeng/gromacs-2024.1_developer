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

#ifndef NBLIB_LISTEDFORCES_LISTEDTESTHELPERS_H
#define NBLIB_LISTEDFORCES_LISTEDTESTHELPERS_H

#include "gromacs/math/vectypes.h"

#include "nblib/listed_forces/definitions.h"

namespace nblib
{
class Box;

//! \brief Creates a default vector of indices for two-centered interactions
template<class Interaction, std::enable_if_t<Contains<Interaction, SupportedTwoCenterTypes>{}>* = nullptr>
std::vector<InteractionIndex<Interaction>> indexVector()
{
    return { { 0, 1, 0 } };
}

//! \brief Creates a default vector of indices for three-centered interactions
template<class Interaction, std::enable_if_t<Contains<Interaction, SupportedThreeCenterTypes>{}>* = nullptr>
std::vector<InteractionIndex<Interaction>> indexVector()
{
    return { { 0, 1, 2, 0 } };
}

//! \brief Creates a default vector of indices for four-centered interactions
template<class Interaction, std::enable_if_t<Contains<Interaction, SupportedFourCenterTypes>{}>* = nullptr>
std::vector<InteractionIndex<Interaction>> indexVector()
{
    return { { 0, 1, 2, 3, 0 } };
}

//! \brief Sets up the calculation fixtures for both Nblib and GMX and compares the resultant forces
void compareNblibAndGmxListedImplementations(const ListedInteractionData&  interactionData,
                                             const std::vector<gmx::RVec>& coordinates,
                                             size_t                        numParticles,
                                             int                           numThreads,
                                             const Box&                    box,
                                             real                          tolerance);

} // namespace nblib

#endif // NBLIB_LISTEDFORCES_LISTEDTESTHELPERS_H
