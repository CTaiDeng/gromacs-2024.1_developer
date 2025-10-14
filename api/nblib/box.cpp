/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Implements nblib simulation box
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "nblib/box.h"

#include <cmath>

#include <algorithm>

#include "nblib/exception.h"

namespace nblib
{

Box::Box(real l) : Box(l, l, l) {}

Box::Box(real x, real y, real z) : legacyMatrix_{ { 0 } }
{
    if (std::isnan(x) || std::isinf(x) || std::isnan(y) || std::isinf(y) || std::isnan(z) || std::isinf(z))
    {
        throw InputException("Cannot have NaN or Inf box length.");
    }

    legacyMatrix_[dimX][dimX] = x;
    legacyMatrix_[dimY][dimY] = y;
    legacyMatrix_[dimZ][dimZ] = z;
}

bool operator==(const Box& rhs, const Box& lhs)
{
    using real_ptr = const real*;
    return std::equal(real_ptr(rhs.legacyMatrix()),
                      real_ptr(rhs.legacyMatrix()) + dimSize * dimSize,
                      real_ptr(lhs.legacyMatrix()));
}

} // namespace nblib
