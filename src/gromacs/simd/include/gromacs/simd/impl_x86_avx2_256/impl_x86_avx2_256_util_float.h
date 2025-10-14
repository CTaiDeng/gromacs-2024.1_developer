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

#ifndef GMX_SIMD_IMPL_X86_AVX2_256_UTIL_FLOAT_H
#define GMX_SIMD_IMPL_X86_AVX2_256_UTIL_FLOAT_H

#include "config.h"

#include <immintrin.h>

#include "gromacs/simd/impl_x86_avx_256/impl_x86_avx_256_util_float.h"

namespace gmx
{

// This version is marginally slower than the AVX 4-wide component
// load version on Intel Skylake. On older Intel architectures this
// version is significantly slower. However, the code using the
// intrinsic used here is understood better by TSAN and can be used to
// test GROMACS code in an AVX2_256 (or higher) build configuration
// together with TSAN without false positives. Otherwise, the LINCS
// code reports correct races that are in fact benign because the
// values upon which the race occurs are only used in the load and
// subsequent transpose and not for any computation.
template<int align>
static inline void gmx_simdcall gatherLoadUTransposeSafe(const float*       base,
                                                         const std::int32_t offset[],
                                                         SimdFloat*         v0,
                                                         SimdFloat*         v1,
                                                         SimdFloat*         v2)
{
    assert(std::size_t(offset) % 32 == 0);

    const SimdFInt32 alignSimd = SimdFInt32(align);

    SimdFInt32 vindex = simdLoad(offset, SimdFInt32Tag());
    vindex            = vindex * alignSimd;

    *v0 = _mm256_i32gather_ps(base + 0, vindex.simdInternal_, sizeof(float));
    *v1 = _mm256_i32gather_ps(base + 1, vindex.simdInternal_, sizeof(float));
    *v2 = _mm256_i32gather_ps(base + 2, vindex.simdInternal_, sizeof(float));
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_AVX2_256_UTIL_FLOAT_H
