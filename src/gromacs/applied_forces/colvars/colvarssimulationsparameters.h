/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2023- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Declares parameters needed during simulation time
 *
 * \author Hubert Santuz <hubert.santuz@gmail.com>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_COLVARSIMULATIONSPARAMETERS_H
#define GMX_APPLIED_FORCES_COLVARSIMULATIONSPARAMETERS_H

#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/utility/logger.h"

namespace gmx
{

/*! \internal
 * \brief Collect colvars parameters only available during simulation setup.
 *
 * To build the colvars force provider during simulation setup,
 * one needs access to parameters that become available only during simulation setup.
 *
 * This class collects these parameters via MdModuleNotifications in the
 * simulation setup phase and provides a check if all necessary parameters have
 * been provided.
 */
class ColvarsSimulationsParameters
{
public:
    ColvarsSimulationsParameters() = default;

    //! Set the local atom set Manager for colvars.
    void setLocalAtomSetManager(LocalAtomSetManager* localAtomSetManager);
    //! Get the local atom set Manager for colvars.
    LocalAtomSetManager* localAtomSetManager() const;


    /*! \brief Construct the topology of the system.
     *
     * \param[in] mtop is the pointer to the global topology struct
     */
    void setTopology(const gmx_mtop_t& mtop);

    //! Get the topology
    t_atoms topology() const;

    /*! \brief Set the periodic boundary condition via MdModuleNotifier.
     *
     * The pbc type is wrapped in PeriodicBoundaryConditionType to
     * allow the MdModuleNotifier to statically distinguish the callback
     * function type from other 'int' function callbacks.
     *
     * \param[in] pbcType enumerates the periodic boundary condition.
     */
    void setPeriodicBoundaryConditionType(const PbcType& pbcType);

    //! Get the periodic boundary conditions
    PbcType periodicBoundaryConditionType();

    //! Set the simulation time step
    void setSimulationTimeStep(double timeStep);
    //! Return the simulation time step
    double simulationTimeStep() const;

    //! Set the communicator
    void setComm(const t_commrec& cr);
    //! Return the communicator
    const t_commrec* comm() const;

    /*! \brief Set the logger for QMMM during mdrun
     * \param[in] logger Logger instance to be used for output
     */
    void setLogger(const MDLogger& logger);

    //! Get the logger instance
    const MDLogger* logger() const;

private:
    //! The LocalAtomSetManager
    LocalAtomSetManager* localAtomSetManager_;
    //! The type of periodic boundary conditions in the simulation
    std::unique_ptr<PbcType> pbcType_;
    //! The simulation time step
    double simulationTimeStep_ = 1;
    //! The topology
    t_atoms gmxAtoms_;
    //! The communicator
    const t_commrec* cr_;
    //! MDLogger for notifications during mdrun
    const MDLogger* logger_ = nullptr;


    // GMX_DISALLOW_COPY_AND_ASSIGN(ColvarsSimulationsParameters);
};

} // namespace gmx

#endif
