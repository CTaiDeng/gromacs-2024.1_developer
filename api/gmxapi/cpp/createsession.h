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

#ifndef GMXAPI_LIBRARY_SESSION_H
#define GMXAPI_LIBRARY_SESSION_H
/*! \file
 * \brief Library internal details for Session API class(es).
 *
 * \ingroup gmxapi
 */

#include "gromacs/mdrunutility/logging.h"

// Above are headers for dependencies.
// Following are public headers for the current module.
#include "gmxapi/context.h"
#include "gmxapi/session.h"

namespace gmx
{
class MdrunnerBuilder;
class SimulationContext;
} // end namespace gmx

namespace gmxapi
{

/*!
 * \brief Factory free function for creating new Session objects.
 *
 * \param context Shared ownership of a Context implementation instance.
 * \param runnerBuilder MD simulation builder to take ownership of.
 * \param simulationContext Take ownership of the simulation resources.
 * \param logFilehandle Take ownership of filehandle for MD logging
 * \param multiSim Take ownership of resources for Mdrunner multi-sim.
 *
 * \todo Log file management will be updated soon.
 *
 * \return Ownership of new Session implementation instance.
 */
std::shared_ptr<Session> createSession(std::shared_ptr<ContextImpl> context,
                                       gmx::MdrunnerBuilder&&       runnerBuilder,
                                       gmx::SimulationContext&&     simulationContext,
                                       gmx::LogFilePtr              logFilehandle);


} // end namespace gmxapi

#endif // GMXAPI_LIBRARY_SESSION_H
