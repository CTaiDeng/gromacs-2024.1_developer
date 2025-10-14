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

#ifndef GMX_GMXPREPROCESS_TOPDIRS_H
#define GMX_GMXPREPROCESS_TOPDIRS_H

/* Must correspond to strings in topdirs.cpp */
enum class Directive : int
{
    d_defaults,
    d_atomtypes,
    d_bondtypes,
    d_constrainttypes,
    d_pairtypes,
    d_angletypes,
    d_dihedraltypes,
    d_nonbond_params,
    d_implicit_genborn_params,
    d_implicit_surface_params,
    d_cmaptypes,
    d_moleculetype,
    d_atoms,
    d_vsites1,
    d_vsites2,
    d_vsites3,
    d_vsites4,
    d_vsitesn,
    d_bonds,
    d_exclusions,
    d_pairs,
    d_pairs_nb,
    d_angles,
    d_dihedrals,
    d_constraints,
    d_settles,
    d_polarization,
    d_water_polarization,
    d_thole_polarization,
    d_system,
    d_molecules,
    d_position_restraints,
    d_angle_restraints,
    d_angle_restraints_z,
    d_distance_restraints,
    d_orientation_restraints,
    d_dihedral_restraints,
    d_cmap,
    d_intermolecular_interactions,
    d_maxdir,
    d_invalid,
    d_none,
    Count
};

const char* enumValueToString(Directive d);

struct DirStack
{
    Directive d;
    DirStack* prev;
};

int ifunc_index(Directive d, int type);

Directive str2dir(char* dstr);

void DS_Init(DirStack** DS);

void DS_Done(DirStack** DS);

void DS_Push(DirStack** DS, Directive d);

int DS_Search(DirStack* DS, Directive d);

int DS_Check_Order(DirStack* DS, Directive d);

#endif
