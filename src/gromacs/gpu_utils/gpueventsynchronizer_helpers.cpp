/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2022- The GROMACS Authors
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

/*! \libinternal \file
 *  \brief Helper functions for a GpuEventSynchronizer class.
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 * \inlibraryapi
 */

#include "gmxpre.h"

#include "gpueventsynchronizer_helpers.h"

#include "config.h"

#include "gromacs/utility/gmxassert.h"

#if GMX_GPU_CUDA
// Enable event consumption tracking in debug builds, see #3988.
// In OpenCL and SYCL builds, g_useEventConsumptionCounting is constexpr true.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern bool g_useEventConsumptionCounting;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool g_useEventConsumptionCounting = (CMAKE_BUILD_TYPE == CMAKE_BUILD_TYPE_DEBUG
                                      || CMAKE_BUILD_TYPE == CMAKE_BUILD_TYPE_RELWITHDEBINFO);
#endif

namespace gmx::internal
{
void disableCudaEventConsumptionCounting()
{
    GMX_RELEASE_ASSERT(GMX_GPU_CUDA != 0, "Can only be called in CUDA builds");
#if GMX_GPU_CUDA
    /* With threadMPI, we can have a race between different threads setting and reading this flag.
     * However, either all ranks call this function or no one does,
     * so the expected value is the same for all threads,
     * and each thread reads the flag only after callin this function (or deciding no to),
     * so we cannot have any inconsistencies. */
    g_useEventConsumptionCounting = false;
#endif
}
} // namespace gmx::internal
