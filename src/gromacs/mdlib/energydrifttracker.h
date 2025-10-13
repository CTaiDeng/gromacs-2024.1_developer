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
 * \brief Declares and defines the EnergyDriftTracker class.
 *
 * \author Berk Hess <hess@kth.se>
 *
 * \ingroup module_mdlib
 */
#ifndef GMX_MDLIB_ENERGYDRIFTTRACKER_H
#define GMX_MDLIB_ENERGYDRIFTTRACKER_H

#include <string>

#include "gromacs/utility/real.h"

namespace gmx
{

/*! \internal
 * \brief Class for tracking and printing the drift in the conserved energy quantity
 */
class EnergyDriftTracker
{
public:
    /*! \brief Constructor
     *
     * \param[in] numAtoms  The total number of atoms in the system
     */
    EnergyDriftTracker(int numAtoms) : numAtoms_(numAtoms) {}

    //! Add a point to the conserved energy tracking
    void addPoint(double time, double energy);

    //! Returns the time of the last point minus the time of the first point
    double timeInterval() const { return lastTime_ - firstTime_; }

    //! Returns the energy drift over the measured interval
    double energyDrift() const;

    /*! \brief Returns two-line string with the time interval and drift over the interval
     *
     * \param[in] partName  A descriptive name for the period over which the tracking occurred
     */
    std::string energyDriftString(const std::string& partName) const;

private:
    //! Whether we stored the first point
    bool storedFirst_ = false;
    //! The first time stored
    double firstTime_ = 0;
    //! The energy for the first time point
    double firstEnergy_ = 0;
    //! The last time stored
    double lastTime_ = 0;
    //! The energy for the last time point
    double lastEnergy_ = 0;
    //! The number of atoms in the system
    int numAtoms_;
};

} // namespace gmx

#endif
