/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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
 * Implements functions from cmdlineinit.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_commandline
 */
#include "gmxpre.h"

#include "cmdlineinit.h"

#include <cstring>

#include <memory>
#include <utility>

#include "gromacs/commandline/cmdlinemodulemanager.h"
#include "gromacs/commandline/cmdlineoptionsmodule.h"
#include "gromacs/commandline/cmdlineprogramcontext.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/datafilefinder.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/init.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/smalloc.h"

namespace gmx
{

namespace
{

//! \addtogroup module_commandline
//! \{

// These never release ownership.
//! Global context instance initialized in initForCommandLine().
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unique_ptr<CommandLineProgramContext> g_commandLineContext;
//! Global library data file finder that respects GMXLIB.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unique_ptr<DataFileFinder> g_libFileFinder;

/*! \brief
 * Broadcasts given data from rank zero to all other ranks.
 */
void broadcastWorld(int size, void* buffer)
{
#if GMX_MPI
    MPI_Bcast(buffer, size, MPI_BYTE, 0, MPI_COMM_WORLD);
#else
    GMX_UNUSED_VALUE(size);
    GMX_UNUSED_VALUE(buffer);
#endif
}

/*! \brief
 * Broadcasts command-line arguments to all ranks.
 *
 * MPI does not ensure that command-line arguments would be passed on any
 * other rank than zero, but our code wants to parse them on each rank
 * separately.
 */
void broadcastArguments(int* argc, char*** argv)
{
    if (gmx_node_num() <= 1)
    {
        return;
    }
    broadcastWorld(sizeof(*argc), argc);

    const bool isMain = (gmx_node_rank() == 0);
    if (!isMain)
    {
        snew(*argv, *argc + 1);
    }
    for (int i = 0; i < *argc; i++)
    {
        int len;
        if (isMain)
        {
            len = std::strlen((*argv)[i]) + 1;
        }
        broadcastWorld(sizeof(len), &len);
        if (!isMain)
        {
            snew((*argv)[i], len);
        }
        broadcastWorld(len, (*argv)[i]);
    }
}

//! \}

} // namespace

CommandLineProgramContext& initForCommandLine(int* argc, char*** argv)
{
    gmx::init(argc, argv);
    GMX_RELEASE_ASSERT(!g_commandLineContext, "initForCommandLine() calls cannot be nested");
    // TODO: Consider whether the argument broadcast would better be done
    // in CommandLineModuleManager.
    broadcastArguments(argc, argv);
    try
    {
        g_commandLineContext = std::make_unique<CommandLineProgramContext>(*argc, *argv);
        setProgramContext(g_commandLineContext.get());
        g_libFileFinder = std::make_unique<DataFileFinder>();
        g_libFileFinder->setSearchPathFromEnv("GMXLIB");
        setLibraryFileFinder(g_libFileFinder.get());
    }
    catch (const std::exception& ex)
    {
        printFatalErrorMessage(stderr, ex);
        std::exit(processExceptionAtExit(ex));
    }
    return *g_commandLineContext;
}

void finalizeForCommandLine()
{
    gmx::finalize();
    setLibraryFileFinder(nullptr);
    g_libFileFinder.reset();
    setProgramContext(nullptr);
    g_commandLineContext.reset();
}

int processExceptionAtExitForCommandLine(const std::exception& ex)
{
    int rc = processExceptionAtExit(ex); // Currently this aborts for real MPI
    finalizeForCommandLine();            // thus this MPI_Finalize doesn't matter.
    return rc;
}

int runCommandLineModule(int argc, char* argv[], ICommandLineModule* module)
{
    return CommandLineModuleManager::runAsMainSingleModule(argc, argv, module);
}

int runCommandLineModule(int                                                         argc,
                         char*                                                       argv[],
                         const char*                                                 name,
                         const char*                                                 description,
                         std::function<std::unique_ptr<ICommandLineOptionsModule>()> factory)
{
    return ICommandLineOptionsModule::runAsMain(argc, argv, name, description, std::move(factory));
}

} // namespace gmx

int gmx_run_cmain(int argc, char* argv[], int (*mainFunction)(int, char*[]))
{
    return gmx::CommandLineModuleManager::runAsMainCMain(argc, argv, mainFunction);
}
