/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

#include "simulationinput.h"

#include "gromacs/fileio/checkpoint.h"
#include "gromacs/fileio/tpxio.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/observableshistory.h"
#include "gromacs/mdtypes/state.h"

namespace gmx
{

void applyGlobalSimulationState(const SimulationInput&      simulationInput,
                                PartialDeserializedTprFile* partialDeserializedTpr,
                                t_state*                    globalState,
                                t_inputrec*                 inputRecord,
                                gmx_mtop_t*                 molecularTopology)
{
    *partialDeserializedTpr = read_tpx_state(
            simulationInput.tprFilename_.c_str(), inputRecord, globalState, molecularTopology);
}

void applyLocalState(const SimulationInput&         simulationInput,
                     t_fileio*                      logfio,
                     const t_commrec*               cr,
                     int*                           dd_nc,
                     t_inputrec*                    inputRecord,
                     t_state*                       state,
                     ObservablesHistory*            observablesHistory,
                     bool                           reproducibilityRequested,
                     const MDModulesNotifiers&      mdModulesNotifiers,
                     gmx::ReadCheckpointDataHolder* modularSimulatorCheckpointData,
                     const bool                     useModularSimulator)
{
    load_checkpoint(simulationInput.cpiFilename_.c_str(),
                    logfio,
                    cr,
                    dd_nc,
                    inputRecord,
                    state,
                    observablesHistory,
                    reproducibilityRequested,
                    mdModulesNotifiers,
                    modularSimulatorCheckpointData,
                    useModularSimulator);
}

} // end namespace gmx
