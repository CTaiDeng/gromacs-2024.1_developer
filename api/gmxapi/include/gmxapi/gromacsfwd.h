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

#ifndef GMXAPI_GROMACSFWD_H
#define GMXAPI_GROMACSFWD_H

/*! \ingroup gmxapi
 * \file
 * \brief Provide forward declarations for symbols in the GROMACS public headers.
 *
 * Basic API clients only need to compile
 * and link against the gmxapi target, but some gmxapi classes use opaque pointers to
 * library classes that are forward-declared here.
 * Client code should not need to include this header directly.
 *
 * For maximal compatibility with other libgmxapi clients (such as third-party
 * Python modules), client code should use the wrappers and protocols in the
 * gmxapi.h header.
 *
 * Note that there is a separate CMake target to build the full
 * developer documentation for gmxapi.
 * Refer to GMXAPI developer docs for the protocols that map gmxapi interfaces to
 * GROMACS library interfaces.
 * Refer to the GROMACS developer
 * documentation for details on library interfaces forward-declared in this header.
 *
 * \todo Improve documentation cross-linking.
 */

// Forward declaration for src/gromacs/mdtypes/inputrec.h
struct t_inputrec;

namespace gmx
{

// Forward declaration for libgromacs header gromacs/restraint/restraintpotential.h
class IRestraintPotential;

} // end namespace gmx

#endif // GMXAPI_GROMACSFWD_H
