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

#ifndef GMX_GMXANA_PRINC_H
#define GMX_GMXANA_PRINC_H

#include "gromacs/math/vectypes.h"
#include "gromacs/topology/idef.h"

struct t_atom;
struct t_atoms;

void rotate_atoms(int gnx, const int index[], rvec x[], matrix trans);
/* Rotate all atoms in index using matrix trans */

void principal_comp(int n, const int index[], t_atom atom[], rvec x[], matrix trans, rvec d);
/* Calculate the principal components of atoms in index. Atoms are
 * mass weighted. It is assumed that the center of mass is in the origin!
 */

void orient_princ(const t_atoms* atoms, int isize, const int* index, int natoms, rvec x[], rvec* v, rvec d);
/* rotates molecule to align principal axes with coordinate axes */

real calc_xcm(const rvec x[], int gnx, const int* index, const t_atom* atom, rvec xcm, bool bQ);
/* Calculate the center of mass of the atoms in index. if bQ then the atoms
 * will be charge weighted rather than mass weighted.
 * Returns the total mass/charge.
 */

real sub_xcm(rvec x[], int gnx, const int* index, const t_atom atom[], rvec xcm, bool bQ);
/* Calc. the center of mass and subtract it from all coordinates.
 * Returns the original center of mass in xcm
 * Returns the total mass
 */

#endif
