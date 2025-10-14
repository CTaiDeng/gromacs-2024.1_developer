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

#include "gmxpre.h"

#include "simulationcontext.h"

#include "config.h"

#include <cassert>

#include "gromacs/mdrunutility/multisim.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxmpi.h"

#include "runner.h"

namespace gmx
{
//! \cond libapi
SimulationContext::SimulationContext(MPI_Comm                    communicator,
                                     ArrayRef<const std::string> multiSimDirectoryNames) :
    libraryWorldCommunicator_(communicator)
{
    GMX_RELEASE_ASSERT((GMX_LIB_MPI && (communicator != MPI_COMM_NULL))
                               || (!GMX_LIB_MPI && (communicator == MPI_COMM_NULL)),
                       "With real MPI, a non-null communicator is required. "
                       "Without it, the communicator must be null.");
    if (multiSimDirectoryNames.empty())
    {
        simulationCommunicator_ = communicator;
    }
    else
    {
        multiSimulation_ = buildMultiSimulation(communicator, multiSimDirectoryNames);
        // Use the communicator resulting from the split for the multi-simulation.
        simulationCommunicator_ = multiSimulation_->simulationComm_;
    }
}

//! \endcond
} // end namespace gmx
