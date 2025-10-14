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

#ifndef GMX_SIMD_IMPL_X86_AVX_512_KNL_SIMD4_FLOAT_H
#define GMX_SIMD_IMPL_X86_AVX_512_KNL_SIMD4_FLOAT_H

#include "config.h"

#include <immintrin.h>

#include "gromacs/simd/impl_x86_avx_512/impl_x86_avx_512_simd4_float.h"

namespace gmx
{

static inline Simd4Float gmx_simdcall rsqrt(Simd4Float x)
{
    return {
#ifndef NDEBUG // for debug mask to the 4 actually used elements to not trigger 1/0 fp exception
        _mm512_castps512_ps128(_mm512_maskz_rsqrt28_ps(avx512Int2Mask(0xF),
                                                       _mm512_castps128_ps512(x.simdInternal_)))
#else
        _mm512_castps512_ps128(_mm512_rsqrt28_ps(_mm512_castps128_ps512(x.simdInternal_)))
#endif
    };
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_AVX_512_KNL_SIMD4_FLOAT_H
