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
 * \brief Implements functions from the EnergyDriftTracker class.
 *
 * \author Berk Hess <hess@kth.se>
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "energydrifttracker.h"

#include <cmath>

#include <string>

#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

void EnergyDriftTracker::addPoint(double time, double energy)
{
    GMX_ASSERT(std::isfinite(energy), "Non-finite energy encountered!");

    if (!storedFirst_)
    {
        firstTime_   = time;
        firstEnergy_ = energy;
        storedFirst_ = true;
    }
    lastTime_   = time;
    lastEnergy_ = energy;
}

double EnergyDriftTracker::energyDrift() const
{
    if (timeInterval() > 0)
    {
        return (lastEnergy_ - firstEnergy_) / (timeInterval() * numAtoms_);
    }
    else
    {
        return 0;
    }
}

std::string EnergyDriftTracker::energyDriftString(const std::string& partName) const
{
    std::string mesg;

    if (timeInterval() > 0)
    {
        mesg = formatString("Energy conservation over %s of length %g ps, time %g to %g ps\n",
                            partName.c_str(),
                            timeInterval(),
                            firstTime_,
                            lastTime_);
        mesg += formatString("  Conserved energy drift: %.2e kJ/mol/ps per atom\n", energyDrift());
    }
    else
    {
        mesg = formatString(
                "Time interval for measuring conserved energy has length 0, time %g to %g ps\n",
                firstTime_,
                lastTime_);
    }

    return mesg;
}

} // namespace gmx
