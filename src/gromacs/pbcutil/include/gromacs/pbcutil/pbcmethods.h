/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

#ifndef GMX_PBCUTIL_PBCMETHODS_H
#define GMX_PBCUTIL_PBCMETHODS_H

struct t_topology;
struct t_block;
struct t_atom;
enum class PbcType : int;

#include "gmxpre.h"

#include "gromacs/math/vec.h"

enum
{
    euSel,
    euRect,
    euTric,
    euCompact,
    euNR
};

void calc_pbc_cluster(int ecenter, int nrefat, t_topology* top, PbcType pbcType, rvec x[], const int index[], matrix box);


void put_molecule_com_in_box(int      unitcell_enum,
                             int      ecenter,
                             t_block* mols,
                             int      natoms,
                             t_atom   atom[],
                             PbcType  pbcType,
                             matrix   box,
                             rvec     x[]);

void put_residue_com_in_box(int     unitcell_enum,
                            int     ecenter,
                            int     natoms,
                            t_atom  atom[],
                            PbcType pbcType,
                            matrix  box,
                            rvec    x[]);

void center_x(int ecenter, rvec x[], matrix box, int n, int nc, const int ci[]);

#endif
