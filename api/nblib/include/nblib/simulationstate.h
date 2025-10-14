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

/*! \inpublicapi \file
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
#ifndef NBLIB_SIMULATIONSTATE_H
#define NBLIB_SIMULATIONSTATE_H

#include <memory>
#include <vector>

#include "nblib/box.h"
#include "nblib/topology.h"
#include "nblib/vector.h"

namespace nblib
{

/*! \libinternal
 * \ingroup nblib
 * \brief Simulation State
 *
 * Simulation state description that serves as a snapshot of the system
 * being analysed. Needed to init an MD program. Allows hot-starting simulations.
 */

class SimulationState final
{
public:
    //! Constructor
    SimulationState(const std::vector<Vec3>& coordinates,
                    const std::vector<Vec3>& velocities,
                    const std::vector<Vec3>& forces,
                    Box                      box,
                    Topology                 topology);

    //! Returns topology of the current state
    const Topology& topology() const;

    //! Returns the box
    Box box() const;

    //! Returns a reference to a (modifiable) vector of particle coordinates
    std::vector<Vec3>& coordinates();

    //! Returns a read-only vector of particle coordinates
    const std::vector<Vec3>& coordinates() const;

    //! Returns a reference to a (modifiable) vector of particle velocities
    std::vector<Vec3>& velocities();

    //! Returns a reference to a (modifiable) vector of forces
    std::vector<Vec3>& forces();

private:
    class Impl;
    std::shared_ptr<SimulationState::Impl> simulationStatePtr_;
};

} // namespace nblib

#endif // NBLIB_SIMULATIONSTATE_H
