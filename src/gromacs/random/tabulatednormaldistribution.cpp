/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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

#include "gmxpre.h"

#include "tabulatednormaldistribution.h"

namespace gmx
{

// MSVC does not handle extern template class members correctly even in MSVC 2015,
// so in that case we have to instantiate in every object using it.
#if !defined(_MSC_VER)
// This is by far the most common version of the normal distribution table,
// so we use this as an extern template specialization to avoid instantiating
// the table in all files using it, unless the user has requested a different
// precision or resolution.
template<>
const std::array<real, 1 << detail::c_TabulatedNormalDistributionDefaultBits>
        TabulatedNormalDistribution<>::c_table_ = TabulatedNormalDistribution<>::makeTable();
#else
// Avoid compiler warnings about no public symbols
void TabulatedNormalDistributionDummy() {}
#endif

} // namespace gmx
