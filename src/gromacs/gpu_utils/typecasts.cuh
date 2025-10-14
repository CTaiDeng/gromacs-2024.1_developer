/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 *  \brief Declare functions to be used to cast CPU types to compatible GPU types.
 *
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 *  \inlibraryapi
 */
#ifndef GMX_GPU_UTILS_TYPECASTS_CUH
#define GMX_GPU_UTILS_TYPECASTS_CUH

#include "gmxpre.h"

#include "gromacs/math/vectypes.h"

/*! \brief Cast RVec buffer to float3 buffer.
 *
 * \param[in] in The RVec buffer to cast.
 *
 * \returns Buffer, casted to float3*.
 */
__forceinline__ static __host__ __device__ float3* asFloat3(gmx::RVec* in)
{
    static_assert(sizeof(in[0]) == sizeof(float3),
                  "Size of the host-side data-type is different from the size of the device-side "
                  "counterpart.");
    return reinterpret_cast<float3*>(in);
}

/*! \brief Cast pointer RVec buffer to a pointer to float3 buffer.
 *
 * \param[in] in The Pointer to RVec buffer to cast.
 *
 * \returns Buffer pointer, casted to float3*.
 */
__forceinline__ static __host__ __device__ float3** asFloat3Pointer(gmx::RVec** in)
{
    static_assert(sizeof((*in)[0]) == sizeof(float3),
                  "Size of the host-side data-type is different from the size of the device-side "
                  "counterpart.");
    return reinterpret_cast<float3**>(in);
}
static inline __host__ __device__ const float3* const* asFloat3Pointer(const gmx::RVec* const* in)
{
    static_assert(sizeof((*in)[0]) == sizeof(float3),
                  "Size of the host-side data-type is different from the size of the device-side "
                  "counterpart.");
    return reinterpret_cast<const float3* const*>(in);
}

#endif // GMX_GPU_UTILS_TYPECASTS_CUH
