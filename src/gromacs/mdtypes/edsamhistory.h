/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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

/*
 * This file contains data types containing essential dynamics and
 * flooding data to be stored in the checkpoint file.
 */

#ifndef GMX_MDLIB_EDSAMHISTORY_H
#define GMX_MDLIB_EDSAMHISTORY_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"

/* Helper structure to be able to make essential dynamics / flooding group(s) whole
 *
 * If one uses essential dynamics or flooding on a group of atoms from
 * more than one molecule, we cannot make this group whole with
 * do_pbc_first_mtop(). We assume that the ED group has the correct PBC
 * representation at the beginning of the simulation and keep track
 * of the shifts to always get it into that representation.
 * For proper restarts from a checkpoint we store the positions of the
 * reference group at the time of checkpoint writing.
 */
typedef struct edsamhistory_t
{
    gmx_bool bFromCpt; // Did we start from a checkpoint file?
    int      nED;      // No. of ED/Flooding data sets, if <1 no ED
    int*     nref;     // No. of atoms in i'th reference structure
    int*     nav;      // Same for average structure
    rvec**   old_sref; // Positions of the reference atoms at the last time step (with correct PBC representation)
    rvec**   old_sref_p; // Pointer to these positions
    rvec**   old_sav;    // Same for the average positions
    rvec**   old_sav_p;  // Pointer to these positions
} edsamhistory_t;

#endif
