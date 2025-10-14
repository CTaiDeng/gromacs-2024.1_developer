/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#ifndef GMX_TOPOLOGY_INVBLOCK_H
#define GMX_TOPOLOGY_INVBLOCK_H

#include <vector>

namespace gmx
{
template<typename>
class ListOfLists;
}

/*! \brief Returns a vector of size \p maxElement + 1 with the list index for every element
 *
 * The index will be -1 when an element is not present in any of the lists.
 * Elements should only appear once and obey 0 <= element <= \p maxElement.
 */
std::vector<int> make_invblock(const gmx::ListOfLists<int>& block, int maxElement);

#endif
