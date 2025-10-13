/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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

/*! \libinternal \file
 *
 * \brief Declares a function that makes the list of links between
 * atoms connected by bonded interactions.
 *
 * \inlibraryapi
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_MAKEBONDEDLINKS_H
#define GMX_DOMDEC_MAKEBONDEDLINKS_H

struct gmx_domdec_t;
struct gmx_mtop_t;

namespace gmx
{
template<typename>
class ArrayRef;
struct AtomInfoWithinMoleculeBlock;
} // namespace gmx

/*! \brief Generate a list of links between atoms that are linked by bonded interactions
 *
 * Also stores whether atoms are linked in \p atomInfoForEachMoleculeBlock.
 */
void makeBondedLinks(gmx_domdec_t*                                   dd,
                     const gmx_mtop_t&                               mtop,
                     gmx::ArrayRef<gmx::AtomInfoWithinMoleculeBlock> atomInfoForEachMoleculeBlock);

#endif
