/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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

/*! \libinternal \file
 *
 * \brief
 * This file contains the definition of a container for history data
 * for simulation observables.
 *
 * The container is used for storing the simulation state data that needs
 * to be written to / read from checkpoint file. This struct should only
 * contain pure observable data. Microstate data should be in t_state.
 * The state of the mdrun machinery is also stored elsewhere.
 *
 * \author Berk Hess
 *
 * \inlibraryapi
 * \ingroup module_mdtypes
 */

#ifndef GMX_MDLIB_OBSERVABLESHISTORY_H
#define GMX_MDLIB_OBSERVABLESHISTORY_H

#include <memory>

class energyhistory_t;
class PullHistory;
struct edsamhistory_t;
struct swaphistory_t;

/*! \libinternal \brief Observables history, for writing/reading to/from checkpoint file
 */
struct ObservablesHistory
{
    //! History for energy observables, used for output only
    std::unique_ptr<energyhistory_t> energyHistory;

    //! History for pulling observables, used for output only
    std::unique_ptr<PullHistory> pullHistory;

    //! Essential dynamics and flooding history
    std::unique_ptr<edsamhistory_t> edsamHistory;

    //! Ion/water position swapping history
    std::unique_ptr<swaphistory_t> swapHistory;

    ObservablesHistory();

    ~ObservablesHistory();
};

#endif
