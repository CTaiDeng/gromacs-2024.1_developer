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

#ifndef GMX_PBCUTIL_RMPBC_H
#define GMX_PBCUTIL_RMPBC_H

#include "gromacs/math/vectypes.h"

class InteractionDefinitions;
struct t_atoms;
struct t_idef;
struct t_trxframe;
enum class PbcType : int;

typedef struct gmx_rmpbc* gmx_rmpbc_t;

gmx_rmpbc_t gmx_rmpbc_init(const InteractionDefinitions& idef, PbcType pbcType, int natoms);

gmx_rmpbc_t gmx_rmpbc_init(const t_idef* idef, PbcType pbcType, int natoms);

void gmx_rmpbc_done(gmx_rmpbc_t gpbc);

void gmx_rmpbc_apply(gmx_rmpbc_t gpbc, int natoms, const matrix box, rvec x[]);
/* Correct coordinates x for atoms within every molecule for the periodic
 * boundary conditions such that every molecule is whole.
 * natoms is the size x and can be smaller than the number
 * of atoms in idef, but should only contain complete molecules.
 * When pbcType=PbcType::Unset, the type of pbc is guessed from the box matrix.
 */

void gmx_rmpbc_copy(gmx_rmpbc_t gpbc, int natoms, const matrix box, rvec x[], rvec x_s[]);
/* As gmx_rmpbc, but outputs in x_s and does not modify x. */

void gmx_rmpbc_trxfr(gmx_rmpbc_t gpbc, struct t_trxframe* fr);
/* As gmx_rmpbc but operates on a t_trxframe data structure. */

void rm_gropbc(const t_atoms* atoms, rvec x[], const matrix box);
/* Simple routine for use in analysis tools that just have a pdb or
 * similar file.
 */

#endif
