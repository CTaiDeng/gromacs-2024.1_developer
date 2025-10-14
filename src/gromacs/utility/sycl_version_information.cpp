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

#include "gmxpre.h"

#include "sycl_version_information.h"

#include "config.h"

#if GMX_GPU_SYCL
#    include "gromacs/gpu_utils/gmxsycl.h"
#endif
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

std::string getSyclCompilerVersion()
{
#if GMX_SYCL_DPCPP
#    ifdef __LIBSYCL_MAJOR_VERSION
    return formatString("%d (libsycl %d.%d.%d)",
                        __SYCL_COMPILER_VERSION,
                        __LIBSYCL_MAJOR_VERSION,
                        __LIBSYCL_MINOR_VERSION,
                        __LIBSYCL_PATCH_VERSION);
#    else
    return formatString("%d", __SYCL_COMPILER_VERSION);
#    endif
#elif GMX_SYCL_HIPSYCL
    return hipsycl::sycl::detail::version_string();
#else
    GMX_THROW(gmx::InternalError("Not implemented for non-SYCL build"));
#endif
}

} // namespace gmx
