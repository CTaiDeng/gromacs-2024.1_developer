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

#ifndef GMX_EWALD_PME_SIMD_H
#define GMX_EWALD_PME_SIMD_H

/* Include the SIMD macro file and then check for support */
#include "gromacs/simd/simd.h"

/* Check if we have 4-wide SIMD macro support */
#if GMX_SIMD4_HAVE_REAL
/* Do PME spread and gather with 4-wide SIMD.
 * NOTE: SIMD is only used with PME order 4 and 5 (which are the most common).
 */
#    define PME_SIMD4_SPREAD_GATHER

#    if GMX_SIMD_HAVE_LOADU && GMX_SIMD_HAVE_STOREU
/* With PME-order=4 on x86, unaligned load+store is slightly faster
 * than doubling all SIMD operations when using aligned load+store.
 */
#        define PME_SIMD4_UNALIGNED
#    endif
#endif

#ifdef PME_SIMD4_SPREAD_GATHER
#    define SIMD4_ALIGNMENT (GMX_SIMD4_WIDTH * sizeof(real))
#else
/* We can use any alignment, apart from 0, so we use 4 reals */
#    define SIMD4_ALIGNMENT (4 * sizeof(real))
#endif

/* Check if we can use SIMD with packs of 4 for gather with order 4 */
#if GMX_SIMD_HAVE_4NSIMD_UTIL_REAL && GMX_SIMD_REAL_WIDTH <= 16
#    define PME_4NSIMD_GATHER 1
#else
#    define PME_4NSIMD_GATHER 0
#endif

#endif
