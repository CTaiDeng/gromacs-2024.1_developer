/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/*! \file
 * \internal \brief
 * Declares gmx::internal::LocalAtomSetDataData.
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \ingroup module_domdec
 */
#include "gmxpre.h"

#include "localatomsetdata.h"

#include <algorithm>
#include <numeric>

#include "gromacs/domdec/ga2la.h"

namespace gmx
{

namespace internal
{

/********************************************************************
 * LocalAtomSetData
 */

LocalAtomSetData::LocalAtomSetData(ArrayRef<const Index> globalIndex) :
    globalIndex_(globalIndex.begin(), globalIndex.end()),
    localIndex_(globalIndex.begin(), globalIndex.end())
{
    collectiveIndex_.resize(localIndex_.size());
    std::iota(collectiveIndex_.begin(), collectiveIndex_.end(), 0);
}

void LocalAtomSetData::setLocalAndCollectiveIndices(const gmx_ga2la_t& ga2la)
{
    /* Loop over all the atom indices of the set to check which ones are local.
     * cf. dd_make_local_group_indices in groupcoord.cpp
     */
    int numAtomsGlobal = globalIndex_.size();

    /* Clear vector without changing capacity,
     * because we expect the size of the vectors to vary little. */
    localIndex_.resize(0);
    collectiveIndex_.resize(0);

    for (int iCollective = 0; iCollective < numAtomsGlobal; iCollective++)
    {
        if (const int* iLocal = ga2la.findHome(globalIndex_[iCollective]))
        {
            /* Save the atoms index in the local atom numbers array */
            /* The atom with this index is a home atom. */
            localIndex_.push_back(*iLocal);

            /* Keep track of where this local atom belongs in the collective index array.
             * This is needed when reducing the local arrays to a collective/global array
             * in communicate_group_positions */
            collectiveIndex_.push_back(iCollective);
        }
    }
}

} // namespace internal

} // namespace gmx
