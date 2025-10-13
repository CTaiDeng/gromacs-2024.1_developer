/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2022- The GROMACS Authors
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
 * \brief Handler for mpi4py without MPI-enabled GROMACS.
 *
 * \ingroup module_python
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 */

#include <mpi.h>

#include "pybind11/pybind11.h"

#include "gmxapi/context.h"
#include "gmxapi/mpi/gmxapi_mpi.h"

#include "mpi_bindings.h"
#include "pycontext.h"

namespace py = pybind11;

namespace gmxpy
{

gmxapi::Context context_from_py_comm(py::object communicator)
{
    MPI_Comm* comm_ptr = get_mpi_comm(communicator);

    int size = 0;
    MPI_Comm_size(*comm_ptr, &size);
    if (size != 1)
    {
        const auto message = std::string(
                                     "Installed GROMACS is not MPI-enabled. Cannot accept "
                                     "communicator of size ")
                             + std::to_string(size) + ".";
        throw ResourceError(message);
    }
    // Note: By default, GROMACS ignores the provided communicator when GMX_LIB_MPI is FALSE.
    // Either here or in the library, we _could_ use the provided communicator to help determine
    // the threading parameters.
    return gmxapi::createContext(*gmxapi::assignResource(*get_mpi_comm(communicator)));
}

} // namespace gmxpy
