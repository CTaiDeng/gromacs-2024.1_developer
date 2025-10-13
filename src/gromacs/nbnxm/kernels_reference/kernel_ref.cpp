/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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

#include "gmxpre.h"

#include "kernel_ref.h"

#include <cassert>
#include <cmath>

#include <algorithm>

#include "gromacs/math/functions.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"

/* Analytical reaction-field kernels */
#define CALC_COUL_RF
#define LJ_CUT
#include "kernel_ref_includes.h"
#undef LJ_CUT
#define LJ_FORCE_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_FORCE_SWITCH
#define LJ_POT_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_POT_SWITCH
#define LJ_EWALD
#define LJ_CUT
#define LJ_EWALD_COMB_GEOM
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_GEOM
#define LJ_EWALD_COMB_LB
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_LB
#undef LJ_CUT
#undef LJ_EWALD
#undef CALC_COUL_RF


/* Tabulated exclusion interaction electrostatics kernels */
#define CALC_COUL_TAB
#define LJ_CUT
#include "kernel_ref_includes.h"
#undef LJ_CUT
#define LJ_FORCE_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_FORCE_SWITCH
#define LJ_POT_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_POT_SWITCH
#define LJ_EWALD
#define LJ_CUT
#define LJ_EWALD_COMB_GEOM
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_GEOM
#define LJ_EWALD_COMB_LB
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_LB
#undef LJ_CUT
#undef LJ_EWALD
/* Twin-range cut-off kernels */
#define VDW_CUTOFF_CHECK
#define LJ_CUT
#include "kernel_ref_includes.h"
#undef LJ_CUT
#define LJ_FORCE_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_FORCE_SWITCH
#define LJ_POT_SWITCH
#include "kernel_ref_includes.h"
#undef LJ_POT_SWITCH
#define LJ_EWALD
#define LJ_CUT
#define LJ_EWALD_COMB_GEOM
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_GEOM
#define LJ_EWALD_COMB_LB
#include "kernel_ref_includes.h"
#undef LJ_EWALD_COMB_LB
#undef LJ_CUT
#undef LJ_EWALD
#undef VDW_CUTOFF_CHECK
#undef CALC_COUL_TAB
