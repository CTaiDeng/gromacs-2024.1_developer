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

/*! \internal \file
 *
 * \brief Declares DD cell-size related functions.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_DOMDEC_CELLSIZES_H
#define GMX_DOMDEC_DOMDEC_CELLSIZES_H

#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/timing/wallcycle.h"

namespace gmx
{
template<typename>
class ArrayRef;
}
struct gmx_ddbox_t;
struct gmx_domdec_comm_t;
struct gmx_domdec_t;

/*! \brief Options for setting up a regular, possibly static load balanced, cell grid geometry */
enum
{
    setcellsizeslbLOCAL,     //!< Set cell sizes locally on each rank
    setcellsizeslbMAIN,      //!< Set cell sizes on main rank only
    setcellsizeslbPULSE_ONLY //!< Only set the communication pulses, not the cell sizes
};

/*! \brief Returns the minimum allowed distance between lower and upper bounds of zones along dimension dim_ind */
real grid_jump_limit(const gmx_domdec_comm_t* comm, real cutoff, int dim_ind);

/*! \brief Sets up an initial, non-staggered grid geometry, possibly using static load balancing
 *
 * The number of communication pulses per dimension is returned in numPulses.
 * When setmode==setcellsizeslbMAIN, the cell boundaries per dimension are
 * returned, otherwise an empty arrayref is returned.
 */
gmx::ArrayRef<const std::vector<real>>
set_dd_cell_sizes_slb(gmx_domdec_t* dd, const gmx_ddbox_t* ddbox, int setmode, ivec numPulses);

/*! \brief General cell size adjustment, possibly applying dynamic load balancing */
void set_dd_cell_sizes(gmx_domdec_t*      dd,
                       const gmx_ddbox_t* ddbox,
                       gmx_bool           bDynamicBox,
                       gmx_bool           bUniform,
                       gmx_bool           bDoDLB,
                       int64_t            step,
                       gmx_wallcycle*     wcycle);

#endif
