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
 *
 * \brief Implements the DeviceContext for OpenCL
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_gpu_utils
 */
#include "gmxpre.h"

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#ifndef DOXYGEN

/*! \brief Copies of values from cl_driver_diagnostics_intel.h,
 * which isn't guaranteed to be available. */
/**@{*/
#    ifndef CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL
#        define CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL 0x4106
#    endif
#    ifndef CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL
#        define CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL 0x1
#    endif
#    ifndef CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL
#        define CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL 0x2
#    endif
#    ifndef CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL
#        define CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL 0x4
#    endif
/**@}*/

DeviceContext::DeviceContext(const DeviceInformation& deviceInfo) : deviceInfo_(deviceInfo)
{
    cl_platform_id                     platformId = deviceInfo.oclPlatformId;
    cl_device_id                       deviceId   = deviceInfo.oclDeviceId;
    std::vector<cl_context_properties> contextProperties;

    contextProperties.emplace_back(CL_CONTEXT_PLATFORM);
    contextProperties.emplace_back(reinterpret_cast<cl_context_properties>(platformId));

    if (getenv("GMX_OCL_SHOW_DIAGNOSTICS"))
    {
        contextProperties.emplace_back(CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL);
        contextProperties.emplace_back(CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL
                                       | CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL);
    }
    contextProperties.emplace_back(0);

    cl_int clError;
    context_ = clCreateContext(contextProperties.data(), 1, &deviceId, nullptr, nullptr, &clError);
    if (clError != CL_SUCCESS)
    {
        GMX_THROW(gmx::InternalError(gmx::formatString(
                "Failed to create OpenCL context on device %s (OpenCL error ID %d).",
                deviceInfo.device_name,
                clError)));
    }
}

DeviceContext::~DeviceContext()
{
    cl_int clError;

    if (context_)
    {
        clError = clReleaseContext(context_);
        GMX_RELEASE_ASSERT(
                clError == CL_SUCCESS,
                gmx::formatString("Failed to release OpenCL context (OpenCL error ID %d).", clError).c_str());
        context_ = nullptr;
    }
}

cl_context DeviceContext::context() const
{
    return context_;
}

#endif
