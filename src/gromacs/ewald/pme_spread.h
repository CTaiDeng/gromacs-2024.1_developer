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

#ifndef GMX_EWALD_PME_SPREAD_H
#define GMX_EWALD_PME_SPREAD_H

#include "gromacs/utility/real.h"

#include "pme_internal.h"

void spread_on_grid(const gmx_pme_t*  pme,
                    PmeAtomComm*      atc,
                    const pmegrids_t* grids,
                    gmx_bool          bCalcSplines,
                    gmx_bool          bSpread,
                    real*             fftgrid,
                    gmx_bool          bDoSplines,
                    int               grid_index);

#endif
