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

#ifndef GMX_GMX_NM2TYPE_H
#define GMX_GMX_NM2TYPE_H

#include <cstdio>

#include <filesystem>

class PreprocessingAtomTypes;
struct t_atoms;
struct InteractionsOfType;
struct t_symtab;

struct t_nm2type
{
    char *  elem, *type;
    double  q, m;
    int     nbonds;
    char**  bond;
    double* blen;
};

t_nm2type* rd_nm2type(const std::filesystem::path& ffdir, int* nnm);
/* Read the name 2 type database. nnm is the number of entries
 * ff is the force field.
 */

void dump_nm2type(FILE* fp, int nnm, t_nm2type nm2t[]);
/* Dump the database for debugging. Can be reread by the program */

int nm2type(int nnm, t_nm2type nm2t[], t_atoms* atoms, PreprocessingAtomTypes* atype, int* nbonds, InteractionsOfType* bond);
/* Try to determine the atomtype (force field dependent) for the atoms
 * with help of the bond list
 */

#endif
