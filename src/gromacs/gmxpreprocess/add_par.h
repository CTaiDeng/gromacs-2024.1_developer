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

#ifndef GMX_GMXPREPROCESS_ADD_PAR_H
#define GMX_GMXPREPROCESS_ADD_PAR_H

#include "gromacs/utility/real.h"

struct InteractionsOfType;
struct PreprocessResidue;

namespace gmx
{
template<typename>
class ArrayRef;
}

void add_param(InteractionsOfType* ps, int ai, int aj, gmx::ArrayRef<const real> c, const char* s);

void add_cmap_param(InteractionsOfType* ps, int ai, int aj, int ak, int al, int am, const char* s);

void add_vsite3_atoms(InteractionsOfType* ps, int ai, int aj, int ak, int al, bool bSwapParity);

void add_vsite2_param(InteractionsOfType* ps, int ai, int aj, int ak, real c0);

void add_vsite3_param(InteractionsOfType* ps, int ai, int aj, int ak, int al, real c0, real c1);

void add_vsite4_atoms(InteractionsOfType* ps, int ai, int aj, int ak, int al, int am);

int search_jtype(const PreprocessResidue& localPpResidue, const char* name, bool bFirstRes);

#endif
