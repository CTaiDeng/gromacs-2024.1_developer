/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/*! \libinternal \file
 *
 * \brief This file declares functions for mdrun to call to make a new
 * domain decomposition, and check it.
 *
 * \author Berk Hess <hess@kth.se>
 * \inlibraryapi
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_PARTITION_H
#define GMX_DOMDEC_PARTITION_H

#include <cstdio>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"

struct gmx_ddbox_t;
struct gmx_domdec_t;
struct gmx_localtop_t;
struct gmx_mtop_t;
struct gmx_wallcycle;
struct pull_t;
struct t_commrec;
struct t_forcerec;
struct t_inputrec;
struct t_nrnb;
class t_state;

namespace gmx
{
template<typename>
class ArrayRef;
class Constraints;
class ForceBuffers;
class ImdSession;
class MDAtoms;
class MDLogger;
struct MDModulesNotifiers;
class VirtualSitesHandler;

//! Check whether the DD grid has moved too far for correctness.
bool check_grid_jump(int64_t step, const gmx_domdec_t* dd, real cutoff, const gmx_ddbox_t* ddbox, bool bFatal);

/*! \brief Print statistics for domain decomposition communication */
void print_dd_statistics(const t_commrec* cr, const t_inputrec& inputrec, FILE* fplog);

/*! \brief Partition the system over the nodes.
 *
 * step is only used for printing error messages.
 * If bMainState==TRUE then state_global from the main node is used,
 * else state_local is redistributed between the nodes.
 * When f!=NULL, *f will be reallocated to the size of state_local.
 *
 * \param[in] fplog         Pointer to the log file
 * \param[in] mdlog         MD file logger
 * \param[in] step          Current step
 * \param[in] cr            Communication record
 * \param[in] bMainState  Is it a main state
 * \param[in] state_global  Global state
 * \param[in] top_global    Global topology
 * \param[in] inputrec      Input record
 * \param[in] mdModulesNotifiers  MDModules notifications handler
 * \param[in] imdSession    IMD handle
 * \param[in] pull_work     Pulling data
 * \param[in] state_local   Local state
 * \param[in] f             Force buffer
 * \param[in] mdAtoms       MD atoms
 * \param[in] top_local     Local topology
 * \param[in] fr            Force record
 * \param[in] vsite         Virtual sites handler
 * \param[in] constr        Constraints
 * \param[in] nrnb          Cycle counters
 * \param[in] wcycle        Timers
 * \param[in] bVerbose      Be verbose
 */
void dd_partition_system(FILE*                     fplog,
                         const gmx::MDLogger&      mdlog,
                         int64_t                   step,
                         const t_commrec*          cr,
                         bool                      bMainState,
                         t_state*                  state_global,
                         const gmx_mtop_t&         top_global,
                         const t_inputrec&         inputrec,
                         const MDModulesNotifiers& mdModulesNotifiers,
                         gmx::ImdSession*          imdSession,
                         pull_t*                   pull_work,
                         t_state*                  state_local,
                         gmx::ForceBuffers*        f,
                         gmx::MDAtoms*             mdAtoms,
                         gmx_localtop_t*           top_local,
                         t_forcerec*               fr,
                         gmx::VirtualSitesHandler* vsite,
                         gmx::Constraints*         constr,
                         t_nrnb*                   nrnb,
                         gmx_wallcycle*            wcycle,
                         bool                      bVerbose);

} // namespace gmx
#endif
