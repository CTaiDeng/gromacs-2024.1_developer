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

#ifndef GMX_SIMD_IMPL_X86_AVX2_128_UTIL_DOUBLE_H
#define GMX_SIMD_IMPL_X86_AVX2_128_UTIL_DOUBLE_H

#include "config.h"

#include <immintrin.h>

#include "gromacs/simd/impl_x86_sse4_1/impl_x86_sse4_1_util_double.h"

namespace gmx
{

static inline void gmx_simdcall expandScalarsToTriplets(SimdDouble  scalar,
                                                        SimdDouble* triplets0,
                                                        SimdDouble* triplets1,
                                                        SimdDouble* triplets2)
{
    triplets0->simdInternal_ = _mm_permute_pd(scalar.simdInternal_, _MM_SHUFFLE2(0, 0));
    triplets1->simdInternal_ = _mm_permute_pd(scalar.simdInternal_, _MM_SHUFFLE2(1, 0));
    triplets2->simdInternal_ = _mm_permute_pd(scalar.simdInternal_, _MM_SHUFFLE2(1, 1));
}

static inline double reduceIncr4ReturnSum(double* m, SimdDouble v0, SimdDouble v1, SimdDouble v2, SimdDouble v3)
{
    __m128d t1, t2, t3, t4;

    t1 = _mm_unpacklo_pd(v0.simdInternal_, v1.simdInternal_);
    t2 = _mm_unpackhi_pd(v0.simdInternal_, v1.simdInternal_);
    t3 = _mm_unpacklo_pd(v2.simdInternal_, v3.simdInternal_);
    t4 = _mm_unpackhi_pd(v2.simdInternal_, v3.simdInternal_);

    t1 = _mm_add_pd(t1, t2);
    t3 = _mm_add_pd(t3, t4);

    assert(std::size_t(m) % 16 == 0);

    t2 = _mm_add_pd(t1, _mm_load_pd(m));
    t4 = _mm_add_pd(t3, _mm_load_pd(m + 2));
    _mm_store_pd(m, t2);
    _mm_store_pd(m + 2, t4);

    t1 = _mm_add_pd(t1, t3);

    t2 = _mm_add_sd(t1, _mm_permute_pd(t1, _MM_SHUFFLE2(1, 1)));
    return *reinterpret_cast<double*>(&t2);
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_AVX2_128_UTIL_DOUBLE_H
