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

#ifndef GMX_SIMD_IMPL_X86_AVX_512_KNL_SIMD_FLOAT_H
#define GMX_SIMD_IMPL_X86_AVX_512_KNL_SIMD_FLOAT_H

#include "config.h"

#include <immintrin.h>

#include "gromacs/math/utilities.h"
#include "gromacs/simd/impl_x86_avx_512/impl_x86_avx_512_simd_float.h"

namespace gmx
{

static inline SimdFloat gmx_simdcall rsqrt(SimdFloat x)
{
    return { _mm512_rsqrt28_ps(x.simdInternal_) };
}

static inline SimdFloat gmx_simdcall rcp(SimdFloat x)
{
    return { _mm512_rcp28_ps(x.simdInternal_) };
}

static inline SimdFloat gmx_simdcall maskzRsqrt(SimdFloat x, SimdFBool m)
{
    return { _mm512_maskz_rsqrt28_ps(m.simdInternal_, x.simdInternal_) };
}

static inline SimdFloat gmx_simdcall maskzRcp(SimdFloat x, SimdFBool m)
{
    return { _mm512_maskz_rcp28_ps(m.simdInternal_, x.simdInternal_) };
}

template<MathOptimization opt = MathOptimization::Safe>
static inline SimdFloat gmx_simdcall exp2(SimdFloat x)
{
    return { _mm512_exp2a23_ps(x.simdInternal_) };
}

template<MathOptimization opt = MathOptimization::Safe>
static inline SimdFloat gmx_simdcall exp(SimdFloat x)
{
    const __m512 argscale    = _mm512_set1_ps(1.44269504088896341F);
    const __m512 invargscale = _mm512_set1_ps(-0.69314718055994528623F);

    if (opt == MathOptimization::Safe)
    {
        // Set the limit to gurantee flush to zero
        const SimdFloat smallArgLimit(-88.f);
        // Since we multiply the argument by 1.44, for the safe version we need to make
        // sure this doesn't result in overflow
        x = max(x, smallArgLimit);
    }

    __m512 xscaled = _mm512_mul_ps(x.simdInternal_, argscale);
    __m512 r       = _mm512_exp2a23_ps(xscaled);

    // exp2a23_ps provides 23 bits of accuracy, but we ruin some of that with our argument
    // scaling. To correct this, we find the difference between the scaled argument and
    // the true one (extended precision arithmetics does not appear to be necessary to
    // fulfill our accuracy requirements) and then multiply by the exponent of this
    // correction since exp(a+b)=exp(a)*exp(b).
    // Note that this only adds two instructions (and maybe some constant loads).

    // find the difference
    x = _mm512_fmadd_ps(invargscale, xscaled, x.simdInternal_);
    // x will now be a _very_ small number, so approximate exp(x)=1+x.
    // We should thus apply the correction as r'=r*(1+x)=r+r*x
    r = _mm512_fmadd_ps(r, x.simdInternal_, r);
    return { r };
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_AVX_512_KNL_SIMD_FLOAT_H
