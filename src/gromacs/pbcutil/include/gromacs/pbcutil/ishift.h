/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#ifndef GMX_PBCUTIL_ISHIFT_H
#define GMX_PBCUTIL_ISHIFT_H

#include "gromacs/math/vectypes.h"

namespace gmx
{
//! Maximum dimensions of grid expressing shifts across PBC
//! \{
constexpr int c_dBoxZ = 1;
constexpr int c_dBoxY = 1;
constexpr int c_dBoxX = 2;
//! \}
namespace detail
{
constexpr int c_nBoxZ    = 2 * gmx::c_dBoxZ + 1;
constexpr int c_nBoxY    = 2 * gmx::c_dBoxY + 1;
constexpr int c_nBoxX    = 2 * gmx::c_dBoxX + 1;
constexpr int c_numIvecs = detail::c_nBoxZ * detail::c_nBoxY * detail::c_nBoxX;
} // namespace detail

constexpr int c_centralShiftIndex = detail::c_numIvecs / 2;
constexpr int c_numShiftVectors   = detail::c_numIvecs;

//! Convert grid coordinates to shift index
static inline int xyzToShiftIndex(int x, int y, int z)
{
    return (detail::c_nBoxX * (detail::c_nBoxY * ((z) + gmx::c_dBoxZ) + (y) + gmx::c_dBoxY) + (x)
            + gmx::c_dBoxX);
}

//! Convert grid coordinates to shift index
static inline int ivecToShiftIndex(ivec iv)
{
    return (xyzToShiftIndex((iv)[XX], (iv)[YY], (iv)[ZZ]));
}

//! Return the shift in the X dimension of grid space corresponding to \c iv
static inline int shiftIndexToXDim(int iv)
{
    return (((iv) % detail::c_nBoxX) - gmx::c_dBoxX);
}
} // namespace gmx
#endif
