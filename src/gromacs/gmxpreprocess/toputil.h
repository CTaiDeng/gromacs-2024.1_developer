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

#ifndef GMX_GMXPREPROCESS_TOPUTIL_H
#define GMX_GMXPREPROCESS_TOPUTIL_H

#include <cstdio>

enum class Directive : int;
class PreprocessingAtomTypes;
struct t_atoms;
struct t_excls;
struct MoleculeInformation;
class InteractionOfType;
struct InteractionsOfType;

namespace gmx
{
template<typename>
class ArrayRef;
}

/* UTILITIES */

void add_param_to_list(InteractionsOfType* list, const InteractionOfType& b);

/* PRINTING */

void print_atoms(FILE* out, PreprocessingAtomTypes* atype, t_atoms* at, int* cgnr, bool bRTPresname);

void print_bondeds(FILE*                                   out,
                   int                                     natoms,
                   Directive                               d,
                   int                                     ftype,
                   int                                     fsubtype,
                   gmx::ArrayRef<const InteractionsOfType> plist);

void print_excl(FILE* out, int natoms, t_excls excls[]);

#endif
