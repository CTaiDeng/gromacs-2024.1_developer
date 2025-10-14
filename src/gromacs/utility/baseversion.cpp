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

#include "gmxpre.h"

#include "gromacs/utility/baseversion.h"

#include "config.h"

#include "gromacs/utility/gmxassert.h"

#include "baseversion_gen.h"

const char* gmx_version()
{
    return gmx_ver_string;
}

const char* gmx_version_git_full_hash()
{
    return gmx_full_git_hash;
}

const char* gmx_version_git_central_base_hash()
{
    return gmx_central_base_hash;
}

const char* gmxDOI()
{
    return gmxSourceDoiString;
}

#if GMX_DOUBLE
void gmx_is_double_precision() {}
#else
void gmx_is_single_precision() {}
#endif

const char* getGpuImplementationString()
{
    // Some flavors of clang complain about unreachable returns.
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wunreachable-code-return"
#endif
    if (GMX_GPU)
    {
        if (GMX_GPU_CUDA)
        {
            return "CUDA";
        }
        else if (GMX_GPU_OPENCL)
        {
            return "OpenCL";
        }
        else if (GMX_GPU_SYCL)
        {
            if (GMX_SYCL_DPCPP)
            {
                return "SYCL (DPCPP)";
            }
            else if (GMX_SYCL_HIPSYCL)
            {
                return "SYCL (hipSYCL)";
            }
            else
            {
                return "SYCL (unknown)";
            }
        }
        else
        {
            GMX_RELEASE_ASSERT(false, "Unknown GPU configuration");
            return "impossible";
        }
    }
    else
    {
        return "disabled";
    }
#ifdef __clang__
#    pragma clang diagnostic pop
#endif
}
