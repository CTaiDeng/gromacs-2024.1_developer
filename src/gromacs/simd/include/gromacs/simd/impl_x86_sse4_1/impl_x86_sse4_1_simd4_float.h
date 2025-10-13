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

#ifndef GMX_SIMD_IMPL_X86_SSE4_1_SIMD4_FLOAT_H
#define GMX_SIMD_IMPL_X86_SSE4_1_SIMD4_FLOAT_H

#include "config.h"

#include <smmintrin.h>

#include "gromacs/simd/impl_x86_sse2/impl_x86_sse2_simd4_float.h"

namespace gmx
{

static inline Simd4Float gmx_simdcall round(Simd4Float x)
{
    return { _mm_round_ps(x.simdInternal_, _MM_FROUND_NINT) };
}

static inline Simd4Float gmx_simdcall trunc(Simd4Float x)
{
    return { _mm_round_ps(x.simdInternal_, _MM_FROUND_TRUNC) };
}

static inline float gmx_simdcall dotProduct(Simd4Float a, Simd4Float b)
{
    __m128 res = _mm_dp_ps(a.simdInternal_, b.simdInternal_, 0x71);
    return *reinterpret_cast<float*>(&res);
}

static inline Simd4Float gmx_simdcall blend(Simd4Float a, Simd4Float b, Simd4FBool sel)
{
    return { _mm_blendv_ps(a.simdInternal_, b.simdInternal_, sel.simdInternal_) };
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_SSE4_1_SIMD4_FLOAT_H
