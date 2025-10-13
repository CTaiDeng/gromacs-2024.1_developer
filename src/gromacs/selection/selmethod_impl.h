/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

#ifndef GMX_SELECTION_SELMETHOD_IMPL_H
#define GMX_SELECTION_SELMETHOD_IMPL_H

#include "selmethod.h"

/*
 * These global variables cannot be const because gmx_ana_selmethod_register()
 * modifies them to set some defaults. This is a small price to pay for the
 * convenience of not having to remember exactly how the selection compiler
 * expects the structures to be filled, and even more so if the expectations
 * change. Also, even if the gmx_ana_selmethod_t structures were made const,
 * the parameters could not be without typecasts somewhere, because the param
 * field in gmx_ana_selmethod_t cannot be declared const.
 *
 * Even though the variables may be modified, this should be thread-safe as
 * modifications are done only in gmx_ana_selmethod_register(), and it should
 * work even if called more than once for the same structure, and even if
 * called concurrently from multiple threads (as long as the selection
 * collection is not the same).
 *
 * All of these problems should go away if/when the selection methods are
 * implemented as C++ classes.
 */

/* From sm_com.c */
extern gmx_ana_selmethod_t sm_cog;
extern gmx_ana_selmethod_t sm_com;
/* From sm_simple.c */
extern gmx_ana_selmethod_t sm_all;
extern gmx_ana_selmethod_t sm_none;
extern gmx_ana_selmethod_t sm_atomnr;
extern gmx_ana_selmethod_t sm_resnr;
extern gmx_ana_selmethod_t sm_resindex;
extern gmx_ana_selmethod_t sm_molindex;
extern gmx_ana_selmethod_t sm_atomname;
extern gmx_ana_selmethod_t sm_pdbatomname;
extern gmx_ana_selmethod_t sm_atomtype;
extern gmx_ana_selmethod_t sm_resname;
extern gmx_ana_selmethod_t sm_insertcode;
extern gmx_ana_selmethod_t sm_chain;
extern gmx_ana_selmethod_t sm_mass;
extern gmx_ana_selmethod_t sm_charge;
extern gmx_ana_selmethod_t sm_altloc;
extern gmx_ana_selmethod_t sm_occupancy;
extern gmx_ana_selmethod_t sm_betafactor;
extern gmx_ana_selmethod_t sm_x;
extern gmx_ana_selmethod_t sm_y;
extern gmx_ana_selmethod_t sm_z;
/* From sm_distance.c */
extern gmx_ana_selmethod_t sm_distance;
extern gmx_ana_selmethod_t sm_mindistance;
extern gmx_ana_selmethod_t sm_within;
/* From sm_insolidangle.c */
extern gmx_ana_selmethod_t sm_insolidangle;
/* From sm_same.c */
extern gmx_ana_selmethod_t sm_same;

/* From sm_merge.c */
extern gmx_ana_selmethod_t sm_merge;
extern gmx_ana_selmethod_t sm_plus;
/* From sm_permute.c */
extern gmx_ana_selmethod_t sm_permute;

#endif
