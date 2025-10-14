/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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

/* TODO find out what this file should be called */
#ifndef GMX_EWALD_PME_GRID_H
#define GMX_EWALD_PME_GRID_H

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct gmx_pme_t;

/*! \brief
 * We allow coordinates to be out the unit-cell by up to 2 box lengths,
 * which might be needed along dimension x for a very skewed unit-cell.
 */
constexpr int c_pmeMaxUnitcellShift = 2;

/*! \brief
 * This affects the size of the lookup table of the modulo operation result,
 * when working with PME local grid indices of the particles.
 */
constexpr int c_pmeNeighborUnitcellCount = 2 * c_pmeMaxUnitcellShift + 1;

struct pmegrid_t;
struct pmegrids_t;

void gmx_sum_qgrid_dd(gmx_pme_t* pme, real* grid, int direction);

int copy_pmegrid_to_fftgrid(const gmx_pme_t* pme, const real* pmegrid, real* fftgrid, int grid_index);

int copy_fftgrid_to_pmegrid(gmx_pme_t* pme, const real* fftgrid, real* pmegrid, int grid_index, int nthread, int thread);

void wrap_periodic_pmegrid(const gmx_pme_t* pme, real* pmegrid);

void unwrap_periodic_pmegrid(gmx_pme_t* pme, real* pmegrid);

void pmegrid_init(pmegrid_t* grid,
                  int        cx,
                  int        cy,
                  int        cz,
                  int        x0,
                  int        y0,
                  int        z0,
                  int        x1,
                  int        y1,
                  int        z1,
                  gmx_bool   set_alignment,
                  int        pme_order,
                  real*      ptr);

void pmegrids_init(pmegrids_t* grids,
                   int         nx,
                   int         ny,
                   int         nz,
                   int         nz_base,
                   int         pme_order,
                   gmx_bool    bUseThreads,
                   int         nthread,
                   int         overlap_x,
                   int         overlap_y);

void pmegrids_destroy(pmegrids_t* grids);

void make_gridindex_to_localindex(int    n,
                                  int    local_start,
                                  int    local_range,
                                  bool   checkRoundingAtBoundary,
                                  int**  global_to_local,
                                  real** fraction_shift);

void set_grid_alignment(int* pmegrid_nz, int pme_order);

void reuse_pmegrids(const pmegrids_t* oldgrid, pmegrids_t* newgrid);

#endif
