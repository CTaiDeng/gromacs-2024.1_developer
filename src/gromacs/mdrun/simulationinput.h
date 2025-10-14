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

/*! \libinternal \file
 * \brief Utilities for interacting with SimulationInput.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 * \ingroup module_mdrun
 * \inlibraryapi
 */

#ifndef GMX_MDRUN_SIMULATIONINPUT_H
#define GMX_MDRUN_SIMULATIONINPUT_H

#include <memory>
#include <string>

#include "gromacs/mdrun/simulationinputhandle.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/checkpointdata.h"

// Forward declarations for types from other modules that are opaque to the public API.
// TODO: Document the sources of these symbols or import a (self-documenting) fwd header.
struct gmx_mtop_t;
struct t_commrec;
struct t_fileio;
struct t_inputrec;
class t_state;
struct ObservablesHistory;
struct PartialDeserializedTprFile;

namespace gmx
{
/*
 * \brief Prescription for molecular simulation.
 *
 * In the first implementation, this is a POD struct to allow removal of direct
 * references to TPR and CPT files from Mdrunner. The interface for SimulationInput
 * should be considered to be *completely unspecified* until resolution of
 * https://gitlab.com/gromacs/gromacs/-/issues/3374
 *
 * Clients should use the utility functions defined in simulationinpututility.h
 *
 * Design note: It is probably sufficient for future versions to compose SimulationInput
 * through a Builder rather than to subclass an Interface or base class. Outside of this
 * translation unit, we should avoid coupling to the class definition until/unless we
 * develop a much better understanding of simulation input portability.
 */
class SimulationInput
{
public:
    SimulationInput(const char* tprFilename, const char* cpiFilename);

    std::string tprFilename_;
    std::string cpiFilename_;
};

/*! \brief Get the global simulation input.
 *
 * Acquire global simulation data structures from the SimulationInput handle.
 * Note that global data is returned in the calling thread. In parallel
 * computing contexts, the client is responsible for calling only where needed.
 *
 * Example:
 *    if (SIMMAIN(cr))
 *    {
 *        // Only the main rank has the global state
 *        globalState = globalSimulationState(simulationInput);
 *
 *        // Read (nearly) all data required for the simulation
 *        applyGlobalInputRecord(simulationInput, inputrec);
 *        applyGlobalTopology(simulationInput, &mtop);
 *     }
 *
 * \todo Factor the logic for global/local and main-rank-checks.
 * The SimulationInput utilities should behave properly for the various distributed data scenarios.
 * Consider supplying data directly to the consumers rather than exposing the
 * implementation details of the legacy aggregate types.
 *
 * \{
 */
// TODO: Remove this monolithic detail as member data can be separately cached and managed. (#3374)
// Note that clients still need tpxio.h for PartialDeserializedTprFile.
void applyGlobalSimulationState(const SimulationInput&      simulationInput,
                                PartialDeserializedTprFile* partialDeserializedTpr,
                                t_state*                    globalState,
                                t_inputrec*                 inputrec,
                                gmx_mtop_t*                 globalTopology);
// TODO: Implement the following, pending further discussion re #3374.
std::unique_ptr<t_state> globalSimulationState(const SimulationInput&);
void                     applyGlobalInputRecord(const SimulationInput&, t_inputrec*);
void                     applyGlobalTopology(const SimulationInput&, gmx_mtop_t*);
//! \}

/*! \brief Initialize local stateful simulation data.
 *
 * Establish an invariant for the simulator at a trajectory point.
 * Call on all ranks (after domain decomposition and task assignments).
 *
 * After this call, the simulator has all of the information it will
 * receive in order to advance a trajectory from the current step.
 * Checkpoint information has been applied, if applicable, and stateful
 * data has been (re)initialized.
 *
 * \warning Mdrunner instances do not clearly distinguish between global and local
 * versions of t_state.
 *
 * \todo Factor the distributed data aspects of simulation input data into the
 *       SimulationInput implementation.
 *
 * \todo Consider refactoring to decouple the checkpoint facility from its consumers
 *       (state, observablesHistory, mdModulesNotifiers, and parts of ir).
 *
 * \warning It is the caller’s responsibility to make sure that
 * preconditions are satisfied for the parameter objects.
 *
 * \see globalSimulationState()
 * \see applyGlobalInputRecord()
 * \see applyGlobalTopology()
 */
void applyLocalState(const SimulationInput&         simulationInput,
                     t_fileio*                      logfio,
                     const t_commrec*               cr,
                     int*                           dd_nc,
                     t_inputrec*                    ir,
                     t_state*                       state,
                     ObservablesHistory*            observablesHistory,
                     bool                           reproducibilityRequested,
                     const MDModulesNotifiers&      notifiers,
                     gmx::ReadCheckpointDataHolder* modularSimulatorCheckpointData,
                     bool                           useModularSimulator);

} // end namespace gmx

#endif // GMX_MDRUN_SIMULATIONINPUT_H
