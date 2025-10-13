/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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

#ifndef GMX_SIMD_IMPL_X86_AVX_128_FMA_SIMD_FLOAT_H
#define GMX_SIMD_IMPL_X86_AVX_128_FMA_SIMD_FLOAT_H

#include "config.h"

#include <immintrin.h>
#include <x86intrin.h>

#include "gromacs/simd/impl_x86_sse4_1/impl_x86_sse4_1_simd_float.h"

namespace gmx
{

static inline float gmx_simdcall reduce(SimdFloat a)
{
    a.simdInternal_ =
            _mm_add_ps(a.simdInternal_, _mm_permute_ps(a.simdInternal_, _MM_SHUFFLE(1, 0, 3, 2)));
    a.simdInternal_ =
            _mm_add_ss(a.simdInternal_, _mm_permute_ps(a.simdInternal_, _MM_SHUFFLE(0, 3, 2, 1)));
    return *reinterpret_cast<float*>(&a);
}

static inline SimdFloat gmx_simdcall fma(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return { _mm_macc_ps(a.simdInternal_, b.simdInternal_, c.simdInternal_) };
}

static inline SimdFloat gmx_simdcall fms(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return { _mm_msub_ps(a.simdInternal_, b.simdInternal_, c.simdInternal_) };
}

static inline SimdFloat gmx_simdcall fnma(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return { _mm_nmacc_ps(a.simdInternal_, b.simdInternal_, c.simdInternal_) };
}

static inline SimdFloat gmx_simdcall fnms(SimdFloat a, SimdFloat b, SimdFloat c)
{
    return { _mm_nmsub_ps(a.simdInternal_, b.simdInternal_, c.simdInternal_) };
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_AVX_128_FMA_SIMD_FLOAT_H
