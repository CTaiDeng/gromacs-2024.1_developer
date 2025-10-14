/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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

/*! \file
 * \brief
 * Declares gmx::TrajectoryAnalysisCommandLineRunner.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inpublicapi
 * \ingroup module_trajectoryanalysis
 */
#ifndef GMX_TRAJECTORYANALYSIS_CMDLINERUNNER_H
#define GMX_TRAJECTORYANALYSIS_CMDLINERUNNER_H

#include <functional>
#include <memory>

#include "gromacs/trajectoryanalysis/analysismodule.h"

namespace gmx
{

class CommandLineModuleManager;
class ICommandLineOptionsModule;

/*! \brief
 * Runner for command-line trajectory analysis tools.
 *
 * This class provides static methods to implement a command-line analysis
 * program, given a TrajectoryAnalysisModule object (or a factory of such).
 * It takes care of common command-line parameters, initializing and evaluating
 * selections, and looping over trajectory frames.
 *
 * \inpublicapi
 * \ingroup module_trajectoryanalysis
 */
class TrajectoryAnalysisCommandLineRunner
{
public:
    /*! \brief
     * Factory method type for creating a trajectory analysis module.
     *
     * This method allows the module creation to be postponed to the point
     * where the module is needed, reducing initialization costs in, e.g.,
     * the `gmx` binary, and simplifying exception handling.
     */
    typedef std::function<TrajectoryAnalysisModulePointer()> ModuleFactoryMethod;

    /*! \brief
     * Implements a main() method that runs a given module.
     *
     * \tparam ModuleType  Trajectory analysis module.
     * \param  argc        \c argc passed to main().
     * \param  argv        \c argv passed to main().
     *
     * This method abstracts away all the logic required to implement a
     * main() method in user tools, allowing that to be changed without
     * requiring changes to the tools themselves.
     *
     * \p ModuleType should be default-constructible and derive from
     * TrajectoryAnalysisModule.
     *
     * Does not throw.  All exceptions are caught and handled internally.
     */
    template<class ModuleType>
    static int runAsMain(int argc, char* argv[])
    {
        return runAsMain(argc, argv, &createModule<ModuleType>);
    }
    /*! \brief
     * Implements a main() method that runs a given module.
     *
     * \param  argc        \c argc passed to main().
     * \param  argv        \c argv passed to main().
     * \param  factory     Function that creates the module on demand.
     *
     * Implements the template runAsMain(), but can also be used
     * independently.
     *
     * Does not throw.  All exceptions are caught and handled internally.
     */
    static int runAsMain(int argc, char* argv[], const ModuleFactoryMethod& factory);
    /*! \brief
     * Registers a command-line module that runs a given module.
     *
     * \param  manager     Manager to register the module to.
     * \param  name        Name of the module to register.
     * \param  description One-line description for the module to register.
     * \param  factory     Function that creates the module on demand.
     *
     * \p name and \p descriptions must be string constants or otherwise
     * stay valid for the duration of the program execution.
     */
    static void registerModule(CommandLineModuleManager*  manager,
                               const char*                name,
                               const char*                description,
                               const ModuleFactoryMethod& factory);
    /*! \brief
     * Create a command-line module that runs the provided analysis module.
     *
     * \param[in]  module     Module to run.
     * \returns    Command-line module that runs the provided analysis
     *      module.
     * \throws std::bad_alloc if out of memory.
     *
     * This is mainly provided for testing purposes that want to bypass
     * CommandLineModuleManager.
     */
    static std::unique_ptr<ICommandLineOptionsModule> createModule(TrajectoryAnalysisModulePointer module);

private:
    // Prevent instantiation.
    TrajectoryAnalysisCommandLineRunner() {}

    /*! \brief
     * Creates a trajectory analysis module of a given type.
     *
     * \tparam ModuleType  Module to create.
     */
    template<class ModuleType>
    static TrajectoryAnalysisModulePointer createModule()
    {
        return TrajectoryAnalysisModulePointer(new ModuleType());
    }
};

} // namespace gmx

#endif
