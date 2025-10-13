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

#ifndef GMX_TOPOLOGY_EXCLUSIONBLOCKS_H
#define GMX_TOPOLOGY_EXCLUSIONBLOCKS_H

#include <vector>

#include "gromacs/utility/arrayref.h"

struct t_blocka;

namespace gmx
{
template<typename>
class ListOfLists;

/*! \libinternal \brief
 * Describes exclusions for a single atom.
 */
struct ExclusionBlock
{
    //! Atom numbers for exclusion.
    std::vector<int> atomNumber;
    //! Number of atoms in the exclusion.
    int nra() const { return atomNumber.size(); }
};

/*! \brief Merge the contents of \c b2 into \c excl.
 *
 * Requires that \c b2 and \c excl describe the same number of
 * particles, if \c b2 describes a non-zero number.
 */
void mergeExclusions(ListOfLists<int>* excl, gmx::ArrayRef<ExclusionBlock> b2);

/*! \brief
 * Convert the exclusions.
 *
 * Convert t_blocka exclusions in \p b into ExclusionBlock form and
 * include them in \p b2.
 *
 * \param[in] b Exclusions in t_blocka form.
 * \param[inout] b2 ExclusionBlocks to populate with t_blocka exclusions.
 */
void blockaToExclusionBlocks(const t_blocka* b, gmx::ArrayRef<ExclusionBlock> b2);

//! Convert the exclusions expressed in \c b into t_blocka form
void exclusionBlocksToBlocka(gmx::ArrayRef<const ExclusionBlock> b2, t_blocka* b);

} // namespace gmx

#endif
