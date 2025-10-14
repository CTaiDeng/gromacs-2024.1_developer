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

#ifndef GMX_TIMING_WALLCYCLEREPORTING_H
#define GMX_TIMING_WALLCYCLEREPORTING_H

/* NOTE: None of the routines here are safe to call within an OpenMP
 * region */

#include <cstdio>

#include <array>

#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/basedefinitions.h"

struct t_commrec;

namespace gmx
{
class MDLogger;
}

struct gmx_wallclock_gpu_nbnxn_t;
struct gmx_wallclock_gpu_pme_t;

using WallcycleCounts = std::array<double, sc_numWallCycleCounters + sc_numWallCycleSubCounters>;
/* Convenience typedef */

WallcycleCounts wallcycle_sum(const t_commrec* cr, gmx_wallcycle* wc);
/* Return a vector of the sum of cycle counts over the nodes in
   cr->mpi_comm_mysim. */

void wallcycle_print(FILE*                            fplog,
                     const gmx::MDLogger&             mdlog,
                     int                              nnodes,
                     int                              npme,
                     int                              nth_pp,
                     int                              nth_pme,
                     double                           realtime,
                     gmx_wallcycle*                   wc,
                     const WallcycleCounts&           cyc_sum,
                     const gmx_wallclock_gpu_nbnxn_t* gpu_nbnxn_t,
                     const gmx_wallclock_gpu_pme_t*   gpu_pme_t);
/* Print the cycle and time accounting */

#endif
