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

#ifndef GMX_SIMD_IMPL_X86_AVX_512_GENERAL_H
#define GMX_SIMD_IMPL_X86_AVX_512_GENERAL_H

#include <immintrin.h>

namespace gmx
{

static inline void simdPrefetch(const void* m)
{
    _mm_prefetch(reinterpret_cast<const char*>(m), _MM_HINT_T0);
}

/*! \brief Return integer from AVX-512 mask
 *
 *  \param m  Mask suitable for use with AVX-512 instructions
 *
 *  \return Short integer representation of mask
 */
static inline short avx512Mask2Int(__mmask16 m)
{
    return static_cast<short>(m);
}

/*! \brief Return AVX-512 mask from integer
 *
 *  \param i  Short integer
 *
 *  \return Mask suitable for use with AVX-512 instructions.
 */
static inline __mmask16 avx512Int2Mask(short i)
{
    return static_cast<__mmask16>(i);
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_AVX_512_GENERAL_H
