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

/* This file is completely threadsafe - keep it that way! */
#include "gmxpre.h"

#include "gromacs/topology/invblock.h"

#include "gromacs/utility/listoflists.h"

std::vector<int> make_invblock(const gmx::ListOfLists<int>& block, const int maxElement)
{
    std::vector<int> invBlock(maxElement + 1, -1);

    for (int listIndex = 0; listIndex < block.ssize(); listIndex++)
    {
        for (const int element : block[listIndex])
        {
            GMX_ASSERT(element >= 0 && element <= maxElement,
                       "List elements should be in range 0 <= element <= maxElement");
            GMX_RELEASE_ASSERT(invBlock[element] == -1, "Double entries in ListOfLists");

            invBlock[element] = listIndex;
        }
    }

    return invBlock;
}
