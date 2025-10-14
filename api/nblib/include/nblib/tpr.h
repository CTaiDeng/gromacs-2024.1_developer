/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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

/*! \inpublicapi \file
 * \brief
 * Implements nblib tpr reading
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#ifndef NBLIB_TPR_H
#define NBLIB_TPR_H

#include <memory>
#include <string>
#include <vector>

#include "nblib/basicdefinitions.h"
#include "nblib/box.h"
#include "nblib/topology.h"
#include "nblib/vector.h"

namespace nblib
{
template<typename T>
struct ExclusionLists;
class Box;

/*! \brief Reads in data from tpr which can be used in construction of force calculators
 *
 *  This object reads data such as topology and coordinates from a tpr file, with the
 *  intent of using that data for construction of nblib force calculator objects. To
 *  allow for maximum flexibility in the calling code, it is the responsibility of the
 *  caller to, e.g., make copies if they want to preserve initial state information or
 *  use the data to construct multiple force calculator objects.
 */
struct TprReader
{
public:
    TprReader(std::string filename);

    //! Particle info where all particles are marked to have Van der Waals interactions
    std::vector<int64_t> particleInteractionFlags_;
    //! particle type id of all particles
    std::vector<int> particleTypeIdOfAllParticles_;
    //! Storage for parameters for short range interactions.
    std::vector<real> nonbondedParameters_;
    //! electrostatic charges
    std::vector<real> charges_;
    //! inverse masses
    std::vector<real> inverseMasses_;
    //! stores information about particles pairs to be excluded from the non-bonded force
    //! calculation exclusion list ranges
    std::vector<int> exclusionListRanges_;
    //! exclusion list elements
    std::vector<int> exclusionListElements_;
    //! coordinates
    std::vector<Vec3> coordinates_;
    //! velocities
    std::vector<Vec3> velocities_;
    //! listed forces data
    ListedInteractionData listedInteractionData_;

    //! bounding box of particle coordinates
    [[nodiscard]] Box getBox() const;

private:
    real boxX_;
    real boxY_;
    real boxZ_;
};

} // namespace nblib
#endif // NBLIB_TPR_H
