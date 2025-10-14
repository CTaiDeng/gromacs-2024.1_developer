/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

#ifndef GMX_GPU_UTILS_GPUTRAITS_OCL_H
#define GMX_GPU_UTILS_GPUTRAITS_OCL_H

/*! \libinternal \file
 *  \brief Declares the OpenCL type traits.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \inlibraryapi
 * \ingroup module_gpu_utils
 */

#include "gromacs/gpu_utils/gmxopencl.h"
#include "gromacs/math/vectypes.h"

using DeviceTexture = void*;

//! \brief Single GPU call timing event
using CommandEvent = cl_event;

//! Convenience alias for 2-wide float
using Float2 = cl_float2;

//! Convenience alias for 3-wide float. Not using cl_float3 due to alignment issues.
using Float3 = gmx::RVec;

//! Convenience alias for 4-wide float.
using Float4 = cl_float4;

/*! \internal \brief
 * GPU kernels scheduling description. This is same in OpenCL/CUDA.
 * Provides reasonable defaults, one typically only needs to set the GPU stream
 * and non-1 work sizes.
 */
struct KernelLaunchConfig
{
    //! Work groups (CUDA blocks) counts
    size_t gridSize[3] = { 1, 1, 1 };
    //! Per work group (CUDA block) thread counts
    size_t blockSize[3] = { 1, 1, 1 };
    //! Shared memory size in bytes
    size_t sharedMemorySize = 0;
};

/*! \brief Sets whether device code can use arrays that are embedded in structs.
 * Note that OpenCL 2.x might be able to do this, but we use 1.2.
 */
#define c_canEmbedBuffers false

#endif
