/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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

/*! \defgroup module_mdrunutility Implementation of mdrun utility functionality
 * \ingroup group_mdrun
 *
 * \brief This module contains code that implements general
 * infrastructure for mdrun that does not suit any other module.
 */
/*! \libinternal \file
 *
 * \brief This file declares functions for mdrun to call to manage the
 * details of doing a restart (ie. reading checkpoints, appending
 * output files).
 *
 * \todo There may be other code in runner.cpp etc. that can usefully
 * live here
 *
 * \author Berk Hess <hess@kth.se>
 * \author Erik Lindahl <erik@kth.se>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \inlibraryapi
 * \ingroup module_mdrunutility
 */

#ifndef GMX_MDRUNUTILITY_HANDLERESTART_H
#define GMX_MDRUNUTILITY_HANDLERESTART_H

#include "gromacs/mdrunutility/logging.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/gmxmpi.h"

struct gmx_multisim_t;
struct t_filenm;

namespace gmx
{

enum class AppendingBehavior;

//! Enumeration for describing how mdrun is (re)starting
enum class StartingBehavior : int
{
    //! Restarting with appending, if a checkpoint is supplied and other conditions are met.
    RestartWithAppending,
    //! Restarting without appending, when a checkpoint is supplied.
    RestartWithoutAppending,
    //! Not restarting
    NewSimulation,
    //! Mark the end of the enumeration
    Count
};

/*! \brief Handle startup of mdrun, particularly regarding -cpi and -append
 *
 * If there is a checkpoint file, then prepare to start from that
 * state. If possible/required, do so with appending. If some files
 * are not found when appending should be done, we will instead issue
 * a fatal error to avoid unintentional problems.
 *
 * If there is no checkpoint file, we return a value to indicate a new
 * simulation is starting.
 *
 * On return, \p fnm is updated with suffix strings for part numbers if we are
 * doing a restart from checkpoint and are not appending.
 *
 * The routine also does communication to coordinate behaviour between
 * all simulations, including for error conditions.
 *
 * \throws FileIOError             When the filesystem behavior prevents the
 *                                 user's choices being implemented.
 * \throws InconsistentInputError  When the users's choices cannot be implemented.
 * \throws GromacsException        On ranks upon which the error condition was
 *                                 not detected.
 *
 * \param[in]    isSimulationMain Whether this rank is the main rank of a simulation
 * \param[in]    communicator       MPI communicator
 * \param[in]    ms                 Handles multi-simulations.
 * \param[in]    appendingBehavior  User choice for appending
 * \param[in]    nfile              Size of fnm struct
 * \param[inout] fnm                Filename parameters to mdrun
 *
 * \return  Description of how mdrun is starting */
std::tuple<StartingBehavior, LogFilePtr> handleRestart(bool                  isSimulationMain,
                                                       MPI_Comm              communicator,
                                                       const gmx_multisim_t* ms,
                                                       AppendingBehavior     appendingBehavior,
                                                       int                   nfile,
                                                       t_filenm              fnm[]);

} // namespace gmx

#endif
