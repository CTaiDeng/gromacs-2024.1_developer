/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

/*! \file
 * \brief Provide a bridge to communication resources appropriate for the library.
 *
 * Define the helper functions that the library provides for to help the client
 * implement the interfaces. (This is dependent on the library configuration.)
 *
 * \author "M. Eric Irrgang <ericirrgang@gmail.com"
 */

#include "gmxapi/mpi/resourceassignment.h"

#include "config.h"

#include "gromacs/utility/gmxmpi.h"

#include "context_impl.h"

namespace gmxapi
{

ResourceAssignment::~ResourceAssignment() = default;

// Base implementation is only overridden by client-provided code in certain
// combinations of library and client build configurations.
void ResourceAssignment::applyCommunicator(CommHandle* dst) const
{
    dst->communicator = MPI_COMM_NULL;
}

#if GMX_LIB_MPI
void offerComm(MPI_Comm src, CommHandle* dst)
{
    dst->communicator = src;
}
#endif

} // end namespace gmxapi
