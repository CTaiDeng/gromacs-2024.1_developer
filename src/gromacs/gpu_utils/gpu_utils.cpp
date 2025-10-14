/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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
 *  \brief Function definitions for non-GPU builds
 *
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 */
#include "gmxpre.h"

#include "gpu_utils.h"

#include "config.h"

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/stringutil.h"

#ifdef _MSC_VER
#    pragma warning(disable : 6237)
#endif

const char* enumValueToString(GpuApiCallBehavior enumValue)
{
    static constexpr gmx::EnumerationArray<GpuApiCallBehavior, const char*> s_gpuApiCallBehaviorNames = {
        "Synchronous", "Asynchronous"
    };
    return s_gpuApiCallBehaviorNames[enumValue];
}

bool decideGpuTimingsUsage()
{
    if (GMX_GPU_CUDA || GMX_GPU_SYCL)
    {
        /* CUDA: timings are incorrect with multiple streams.
         * This is the main reason why they are disabled by default.
         * TODO: Consider turning on by default when we can detect nr of streams.
         *
         * SYCL: compilers and runtimes change rapidly, so we disable timings by default
         * to avoid any possible overhead. */
        return (getenv("GMX_ENABLE_GPU_TIMING") != nullptr);
    }
    else if (GMX_GPU_OPENCL)
    {
        return (getenv("GMX_DISABLE_GPU_TIMING") == nullptr);
    }
    else
    {
        // CPU-only build
        return false;
    }
}
