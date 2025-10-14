/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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


/* This file includes all required (non-)energy flavors of the kernel
 * outer and inner loops, given a Coulomb and VdW treatment.
 */

/* Include the force only kernels */
#include "kernel_ref_outer.h"

/* Include the force+energy kernels */
#define CALC_ENERGIES
#include "kernel_ref_outer.h"
#undef CALC_ENERGIES

/* Include the force+energygroups kernels */
#define CALC_ENERGIES
#define ENERGY_GROUPS
#include "kernel_ref_outer.h"
#undef ENERGY_GROUPS
#undef CALC_ENERGIES
