/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#ifndef GMX_MDLIB_PERF_EST_H
#define GMX_MDLIB_PERF_EST_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"

struct gmx_mtop_t;
struct t_inputrec;

void count_bonded_distances(const gmx_mtop_t& mtop, const t_inputrec& ir, double* ndistance_c, double* ndistance_simd);
/* Count the number of distance calculations in bonded interactions,
 * separately for plain-C and SIMD bonded functions.
 * The computational cost is nearly proportional to the numbers.
 * It is allowed to pass NULL for the last two arguments.
 */

float pme_load_estimate(const gmx_mtop_t& mtop, const t_inputrec& ir, const matrix box);
/* Returns an estimate for the relative load of the PME mesh calculation
 * in the total force calculation.
 * This estimate is reasonable for recent Intel and AMD x86_64 CPUs.
 */

#endif
