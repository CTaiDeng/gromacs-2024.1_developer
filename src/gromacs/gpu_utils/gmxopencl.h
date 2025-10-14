/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * \brief
 * Wraps the complexity of including OpenCL in Gromacs.
 *
 * Because OpenCL 2.0 is not officially supported widely, \Gromacs
 * uses earlier interfaces. Some of those have been deprecated in 2.0,
 * and generate warnings, which we need to suppress.
 *
 * Additionally, this code wraps they way that things work differently
 * on Apple platforms.
 *
 * \inlibraryapi
 */

#ifndef GMX_GPU_UTILS_GMXOPENCL_H
#define GMX_GPU_UTILS_GMXOPENCL_H

/*! \brief Declare to OpenCL SDKs that we intend to use OpenCL API
   features that were deprecated in 1.2 or 2.0, so that they don't
   warn about it. */
///@{
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
///@}
#ifdef __APPLE__
#    include <OpenCL/opencl.h>
#else
#    include <CL/opencl.h>
#endif

#endif
