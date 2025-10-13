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
 * \brief Declares the atom distribution function.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */
#ifndef GMX_DOMDEC_DOMDEC_DISTRIBUTE_H
#define GMX_DOMDEC_DOMDEC_DISTRIBUTE_H

#include "gromacs/utility/basedefinitions.h"

struct df_history_t;
struct gmx_ddbox_t;
struct gmx_domdec_t;
struct gmx_mtop_t;
struct t_block;
class t_state;

namespace gmx
{
class MDLogger;
}

/*! \brief Distributes the state from the main rank to all DD ranks */
void distributeState(const gmx::MDLogger& mdlog,
                     gmx_domdec_t*        dd,
                     const gmx_mtop_t&    mtop,
                     t_state*             state_global,
                     const gmx_ddbox_t&   ddbox,
                     t_state*             state_local);

/*! \brief Distribute the dfhist struct from the main rank to all DD ranks
 *
 * Used by the modular simulator checkpointing
 *
 * \param dd  Domain decomposition information
 * \param dfhist  Free energy history struct
 */
void dd_distribute_dfhist(gmx_domdec_t* dd, df_history_t* dfhist);

#endif
