/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2008- The GROMACS Authors
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

#ifndef GMX_TOPOLOGY_TOPSORT_H
#define GMX_TOPOLOGY_TOPSORT_H

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct gmx_mtop_t;
class InteractionDefinitions;

namespace gmx
{
template<typename>
class ArrayRef;
} // namespace gmx

/* Returns if there are perturbed bonded interactions */
gmx_bool gmx_mtop_bondeds_free_energy(const struct gmx_mtop_t* mtop);

/* Sort all the bonded ilists in idef to have the perturbed ones at the end
 * and set nr_nr_nonperturbed in ilist.
 */
void gmx_sort_ilist_fe(InteractionDefinitions* idef, gmx::ArrayRef<const int64_t> atomInfo);

#endif
