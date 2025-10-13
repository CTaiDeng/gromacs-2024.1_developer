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
 * Implements nblib SimulationState
 *
 * \author Berk Hess <hess@kth.se>
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#include "nblib/simulationstate.h"

#include <memory>
#include <vector>

#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/arrayref.h"

#include "nblib/exception.h"
#include "nblib/util/setup.h"
#include "nblib/vector.h"

#include "simulationstateimpl.h"

namespace nblib
{

SimulationState::SimulationState(const std::vector<Vec3>& coordinates,
                                 const std::vector<Vec3>& velocities,
                                 const std::vector<Vec3>& forces,
                                 Box                      box,
                                 Topology                 topology) :
    simulationStatePtr_(std::make_shared<Impl>(coordinates, velocities, forces, box, topology))
{
}

SimulationState::Impl::Impl(const std::vector<Vec3>& coordinates,
                            const std::vector<Vec3>& velocities,
                            const std::vector<Vec3>& forces,
                            const Box&               box,
                            Topology                 topology) :
    box_(box), topology_(std::move(topology))
{
    auto numParticles = topology_.numParticles();

    if (int(coordinates.size()) != numParticles)
    {
        throw InputException("Coordinates array size mismatch");
    }

    if (int(velocities.size()) != numParticles)
    {
        throw InputException("Velocities array size mismatch");
    }

    if (int(forces.size()) != numParticles)
    {
        throw InputException("Force buffer array size mismatch");
    }

    if (!isRealValued(coordinates))
    {
        throw InputException("Input coordinates has at least one NaN");
    }
    coordinates_ = coordinates;
    if (!isRealValued(velocities))
    {
        throw InputException("Input velocities has at least one NaN");
    }

    velocities_ = velocities;

    forces_ = forces;

    // Ensure that the coordinates are in a box following PBC
    put_atoms_in_box(PbcType::Xyz, box.legacyMatrix(), coordinates_);
}

const Topology& SimulationState::Impl::topology() const
{
    return topology_;
}

Box SimulationState::Impl::box() const
{
    return box_;
}

std::vector<Vec3>& SimulationState::Impl::coordinates()
{
    return coordinates_;
}

std::vector<Vec3>& SimulationState::Impl::velocities()
{
    return velocities_;
}

std::vector<Vec3>& SimulationState::Impl::forces()
{
    return forces_;
}

const Topology& SimulationState::topology() const
{
    return simulationStatePtr_->topology();
}

Box SimulationState::box() const
{
    return simulationStatePtr_->box();
}

std::vector<Vec3>& SimulationState::coordinates()
{
    return simulationStatePtr_->coordinates();
}

const std::vector<Vec3>& SimulationState::coordinates() const
{
    return simulationStatePtr_->coordinates();
}

std::vector<Vec3>& SimulationState::velocities()
{
    return simulationStatePtr_->velocities();
}

std::vector<Vec3>& SimulationState::forces()
{
    return simulationStatePtr_->forces();
}

} // namespace nblib
