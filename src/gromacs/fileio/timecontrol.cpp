/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#include "gmxpre.h"

#include "timecontrol.h"

#include <mutex>
#include <optional>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/real.h"

/* The source code in this file should be thread-safe.
         Please keep it that way. */

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static gmx::EnumerationArray<TimeControl, std::optional<real>> timecontrol = { std::nullopt,
                                                                               std::nullopt,
                                                                               std::nullopt };

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::mutex g_timeControlMutex;

std::optional<real> timeValue(TimeControl tcontrol)
{
    const std::lock_guard<std::mutex> lock(g_timeControlMutex);
    return timecontrol[tcontrol];
}

void setTimeValue(TimeControl tcontrol, real value)
{
    const std::lock_guard<std::mutex> lock(g_timeControlMutex);
    timecontrol[tcontrol].emplace(value);
}

void unsetTimeValue(TimeControl tcontrol)
{
    const std::lock_guard<std::mutex> lock(g_timeControlMutex);
    timecontrol[tcontrol].reset();
}
