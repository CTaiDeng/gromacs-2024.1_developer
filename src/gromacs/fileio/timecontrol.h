/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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

#ifndef GMX_FILEIO_TIMECONTROL_H
#define GMX_FILEIO_TIMECONTROL_H

#include <optional>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

/*! \brief
 * Controls when to start and stop reading trajectory data from files.
 */
enum class TimeControl : int
{
    //! Control setting of start time.
    Begin,
    //! Control setting of final time.
    End,
    //! Control setting of time step.
    Delta,
    //! Maximum number.
    Count
};

//! Return time value if one is set.
std::optional<real> timeValue(TimeControl tcontrol);

/*! \brief
 * Set time value to \p value and set internal state to true.
 *
 * Be aware that this sets the global state of the binary that persists
 * as long as the executable is in memory.
 * To return internal state to its original value, use the unset function.
 *
 * \param[in] tcontrol TimeControl value to change setting for.
 * \param[in] value    The time value to set.
 */
void setTimeValue(TimeControl tcontrol, real value);

/*! \brief
 * Return time value to initial state.
 *
 * Ensures that internal state for \p tcontrol is the same as before
 * using setTimeValue. Useful for combining several tools together.
 */
void unsetTimeValue(TimeControl tcontrol);

#endif
