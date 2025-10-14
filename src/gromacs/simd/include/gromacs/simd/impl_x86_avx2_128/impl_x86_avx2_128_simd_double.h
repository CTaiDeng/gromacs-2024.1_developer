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

#ifndef GMX_SIMD_IMPL_X86_AVX2_128_SIMD_DOUBLE_H
#define GMX_SIMD_IMPL_X86_AVX2_128_SIMD_DOUBLE_H

#include "config.h"

#include <immintrin.h>

#include "gromacs/simd/impl_x86_sse4_1/impl_x86_sse4_1_simd_double.h"

namespace gmx
{

static inline double gmx_simdcall reduce(SimdDouble a)
{
    __m128d b = _mm_add_sd(a.simdInternal_, _mm_permute_pd(a.simdInternal_, _MM_SHUFFLE2(1, 1)));
    return *reinterpret_cast<double*>(&b);
}

static inline SimdDouble gmx_simdcall fma(SimdDouble a, SimdDouble b, SimdDouble c)
{
    return { _mm_fmadd_pd(a.simdInternal_, b.simdInternal_, c.simdInternal_) };
}

static inline SimdDouble gmx_simdcall fms(SimdDouble a, SimdDouble b, SimdDouble c)
{
    return { _mm_fmsub_pd(a.simdInternal_, b.simdInternal_, c.simdInternal_) };
}

static inline SimdDouble gmx_simdcall fnma(SimdDouble a, SimdDouble b, SimdDouble c)
{
    return { _mm_fnmadd_pd(a.simdInternal_, b.simdInternal_, c.simdInternal_) };
}

static inline SimdDouble gmx_simdcall fnms(SimdDouble a, SimdDouble b, SimdDouble c)
{
    return { _mm_fnmsub_pd(a.simdInternal_, b.simdInternal_, c.simdInternal_) };
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_AVX2_128_SIMD_DOUBLE_H
