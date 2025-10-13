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

#ifndef GMX_MDLIB_SPLITTER_H
#define GMX_MDLIB_SPLITTER_H

#include <cstdio>

#include "gromacs/utility/basedefinitions.h"

class InteractionDefinitions;

namespace gmx
{
template<typename>
class ListOfLists;
}

gmx::ListOfLists<int> gen_sblocks(FILE* fp, int at_end, const InteractionDefinitions& idef, bool useSettles);
/* Generate shake blocks from the constraint list. Set useSettles to yes for shake
 * blocks including settles. You normally do not want this.
 */

#endif
