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

#ifndef GMX_SIMD_IMPL_REFERENCE_GENERAL_H
#define GMX_SIMD_IMPL_REFERENCE_GENERAL_H

#include "gromacs/utility/basedefinitions.h"

/*! \libinternal \file
 *
 * \brief Reference SIMD implementation, general utility functions
 *
 * \author Erik Lindahl <erik.lindahl@scilifelab.se>
 *
 * \ingroup module_simd
 */

namespace gmx
{

/*! \brief Prefetch memory at address m
 *
 *  This typically prefetches one cache line of memory from address m,
 *  usually 64bytes or more, but the exact amount will depend on the
 *  implementation. On many platforms this is simply a no-op. Technically it
 *  might not be part of the SIMD instruction set, but since it is a
 *  hardware-specific function that is normally only used in tight loops where
 *  we also apply SIMD, it fits well here.
 *
 *  There are no guarantees about the level of cache or temporality, but
 *  usually we expect stuff to end up in level 2, and be used in a few hundred
 *  clock cycles, after which it stays in cache until evicted (normal caching).
 *
 * \param m Pointer to location prefetch. There are no alignment requirements,
 *        but if the pointer is not aligned the prefetch might start at the
 *        lower cache line boundary (meaning fewer bytes are prefetched).
 */
static inline void gmx_unused simdPrefetch(void gmx_unused* m)
{
    // Do nothing for reference implementation
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_REFERENCE_GENERAL_H
