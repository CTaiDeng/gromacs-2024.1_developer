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

/*! \libinternal \file
 *  \brief Helper functions for a GpuEventSynchronizer class.
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 * \inlibraryapi
 */
#ifndef GMX_GPU_UTILS_GPUEVENTSYNCHRONIZER_HELPERS_H
#define GMX_GPU_UTILS_GPUEVENTSYNCHRONIZER_HELPERS_H

#include "config.h"

namespace gmx::internal
{
//! Setter for \c g_useEventConsumptionCounting.
void disableCudaEventConsumptionCounting();
} // namespace gmx::internal

#endif
