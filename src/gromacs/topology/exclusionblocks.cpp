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

#include "gmxpre.h"

#include "gromacs/topology/exclusionblocks.h"

#include <algorithm>

#include "gromacs/topology/block.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/listoflists.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

namespace
{

//! Converts ListOfLists to a list of ExclusionBlocks
void listOfListsToExclusionBlocks(const ListOfLists<int>& b, gmx::ArrayRef<ExclusionBlock> b2)
{
    for (gmx::Index i = 0; i < b.ssize(); i++)
    {
        for (int jAtom : b[i])
        {
            b2[i].atomNumber.push_back(jAtom);
        }
    }
}

//! Converts a list of ExclusionBlocks to ListOfLists
void exclusionBlocksToListOfLists(gmx::ArrayRef<const ExclusionBlock> b2, ListOfLists<int>* b)
{
    b->clear();

    for (const auto& block : b2)
    {
        b->pushBack(block.atomNumber);
    }
}

} // namespace

void blockaToExclusionBlocks(const t_blocka* b, gmx::ArrayRef<ExclusionBlock> b2)
{
    for (int i = 0; (i < b->nr); i++)
    {
        for (int j = b->index[i]; (j < b->index[i + 1]); j++)
        {
            b2[i].atomNumber.push_back(b->a[j]);
        }
    }
}

void exclusionBlocksToBlocka(gmx::ArrayRef<const ExclusionBlock> b2, t_blocka* b)
{
    int nra = 0;
    int i   = 0;
    for (const auto& block : b2)
    {
        b->index[i] = nra;
        int j       = 0;
        for (const auto& entry : block.atomNumber)
        {
            b->a[nra + j] = entry;
            j++;
        }
        nra += block.nra();
        i++;
    }
    /* terminate list */
    b->index[i] = nra;
}

namespace
{

//! Counts and sorts the exclusions
int countAndSortExclusions(gmx::ArrayRef<ExclusionBlock> b2)
{
    int nra = 0;
    for (auto& block : b2)
    {
        if (block.nra() > 0)
        {
            /* remove double entries */
            std::sort(block.atomNumber.begin(), block.atomNumber.end());
            for (auto atom = block.atomNumber.begin() + 1; atom != block.atomNumber.end();)
            {
                GMX_RELEASE_ASSERT(atom < block.atomNumber.end(),
                                   "Need to stay in range of the size of the blocks");
                auto prev = atom - 1;
                if (*prev == *atom)
                {
                    atom = block.atomNumber.erase(atom);
                }
                else
                {
                    ++atom;
                }
            }
            nra += block.nra();
        }
    }

    return nra;
}

} // namespace

void mergeExclusions(ListOfLists<int>* excl, gmx::ArrayRef<ExclusionBlock> b2)
{
    if (b2.empty())
    {
        return;
    }
    GMX_RELEASE_ASSERT(b2.ssize() == excl->ssize(),
                       "Cannot merge exclusions for "
                       "blocks that do not describe the same number "
                       "of particles");

    /* Convert the t_blocka entries to ExclusionBlock form */
    listOfListsToExclusionBlocks(*excl, b2);

    countAndSortExclusions(b2);

    exclusionBlocksToListOfLists(b2, excl);
}

} // namespace gmx
