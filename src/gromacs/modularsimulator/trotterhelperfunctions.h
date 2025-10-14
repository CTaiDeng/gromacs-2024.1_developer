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

/*! \internal \file
 * \brief Defines helper functions used by the Trotter decomposition
 * algorithms (Nose-Hoover chains, MTTK)
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 *
 * This header is only used within the modular simulator module
 */
#ifndef GMX_MODULARSIMULATOR_TROTTERHELPERFUNCTIONS_H
#define GMX_MODULARSIMULATOR_TROTTERHELPERFUNCTIONS_H

#include "modularsimulatorinterfaces.h"

namespace gmx
{

/*! \brief Check whether two times are nearly equal
 *
 * Times are considered close if their absolute difference is smaller
 * than c_timePrecision.
 *
 * \param time           The test time
 * \param referenceTime  The reference time
 * \return bool          Whether the absolute difference is < c_timePrecision
 */
inline bool timesClose(Time time, Time referenceTime)
{
    /* Expected time precision
     * Times are typically incremented in the order of 1e-3 ps (1 fs), so
     * 1e-6 should be sufficiently tight.
     */
    constexpr real c_timePrecision = 1e-6;

    return (time - referenceTime) * (time - referenceTime) < c_timePrecision * c_timePrecision;
}

} // namespace gmx
#endif // GMX_MODULARSIMULATOR_TROTTERHELPERFUNCTIONS_H
