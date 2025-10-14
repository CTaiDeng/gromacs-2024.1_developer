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

#ifndef GMX_SIMD_IMPL_X86_AVX_512_KNL_SIMD_DOUBLE_H
#define GMX_SIMD_IMPL_X86_AVX_512_KNL_SIMD_DOUBLE_H

#include "config.h"

#include <immintrin.h>

#include "gromacs/simd/impl_x86_avx_512/impl_x86_avx_512_simd_double.h"

namespace gmx
{

static inline SimdDouble gmx_simdcall rsqrt(SimdDouble x)
{
    return { _mm512_rsqrt28_pd(x.simdInternal_) };
}

static inline SimdDouble gmx_simdcall rcp(SimdDouble x)
{
    return { _mm512_rcp28_pd(x.simdInternal_) };
}

static inline SimdDouble gmx_simdcall maskzRsqrt(SimdDouble x, SimdDBool m)
{
    return { _mm512_maskz_rsqrt28_pd(m.simdInternal_, x.simdInternal_) };
}

static inline SimdDouble gmx_simdcall maskzRcp(SimdDouble x, SimdDBool m)
{
    return { _mm512_maskz_rcp28_pd(m.simdInternal_, x.simdInternal_) };
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_AVX_512_KNL_SIMD_DOUBLE_H
