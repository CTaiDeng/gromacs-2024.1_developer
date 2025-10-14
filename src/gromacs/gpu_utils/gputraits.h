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

#ifndef GMX_GPU_UTILS_GPUTRAITS_H
#define GMX_GPU_UTILS_GPUTRAITS_H

/*! \libinternal \file
 *  \brief Declares the GPU type traits for non-GPU builds.
 *
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \inlibraryapi
 * \ingroup module_gpu_utils
 */

#include "config.h"

#if GMX_GPU_CUDA

#    include "gromacs/gpu_utils/gputraits.cuh"

#elif GMX_GPU_OPENCL

#    include "gromacs/gpu_utils/gputraits_ocl.h"

#elif GMX_GPU_SYCL

#    include "gromacs/gpu_utils/gputraits_sycl.h"

#else

using DeviceTexture = void*;

//! \brief Single GPU call timing event
using CommandEvent = void*;

// Stubs for CPU-only build. Might be changed in #3312.
struct Float2
{
};
struct Float3
{
};
struct Float4
{
};

#endif // GMX_GPU

namespace gmx
{
//! Reinterpret-cast any pointer \p in to \c Float3, checking the type compatibility.
template<typename T>
static inline Float3* asGenericFloat3Pointer(T* in)
{
    static_assert(sizeof(T) == sizeof(Float3),
                  "Size of the host-side data-type is different from the size of the generic "
                  "device-side counterpart.");
    return reinterpret_cast<Float3*>(in);
}

//! Reinterpret-cast any const pointer \p in to \c Float3, checking the type compatibility.
template<typename T>
static inline const Float3* asGenericFloat3Pointer(const T* in)
{
    static_assert(sizeof(T) == sizeof(Float3),
                  "Size of the host-side data-type is different from the size of the generic "
                  "device-side counterpart.");
    return reinterpret_cast<const Float3*>(in);
}

//! Reinterpret-cast any container \p in to \c Float3, checking the type compatibility.
template<typename C>
static inline Float3* asGenericFloat3Pointer(C& in)
{
    static_assert(sizeof(*in.data()) == sizeof(Float3),
                  "Size of the host-side data-type is different from the size of the device-side "
                  "counterpart.");
    return reinterpret_cast<Float3*>(in.data());
}

//! Reinterpret-cast any const container \p in to \c Float3, checking the type compatibility.
template<typename C>
static inline const Float3* asGenericFloat3Pointer(const C& in)
{
    static_assert(sizeof(*in.data()) == sizeof(Float3),
                  "Size of the host-side data-type is different from the size of the device-side "
                  "counterpart.");
    return reinterpret_cast<const Float3*>(in.data());
}
} // namespace gmx

#endif // GMX_GPU_UTILS_GPUTRAITS_H
