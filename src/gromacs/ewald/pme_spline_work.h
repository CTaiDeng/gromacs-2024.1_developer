/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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

#ifndef GMX_EWALD_PME_SPLINE_WORK_H
#define GMX_EWALD_PME_SPLINE_WORK_H

#include "gromacs/simd/simd.h"

#include "pme_simd.h"

struct pme_spline_work
{
#ifdef PME_SIMD4_SPREAD_GATHER
    /* Masks for 4-wide SIMD aligned spreading and gathering */
    gmx::Simd4Bool mask_S0[6], mask_S1[6];
#else
    int dummy; /* C89 requires that struct has at least one member */
#endif
};

pme_spline_work* make_pme_spline_work(int order);

void destroy_pme_spline_work(pme_spline_work* work);

#endif
