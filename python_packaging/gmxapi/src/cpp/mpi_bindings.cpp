/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2022- The GROMACS Authors
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
 * \brief Handlers for MPI details.
 *
 * \ingroup module_python
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 */

#include "mpi_bindings.h"

#include "mpi4py/mpi4py.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "gmxpy_exceptions.h"


namespace py = pybind11;

namespace gmxpy
{

MPI_Comm* get_mpi_comm(pybind11::object communicator)
{
    py::gil_scoped_acquire lock;

    MPI_Comm* comm_ptr = PyMPIComm_Get(communicator.ptr());

    if (comm_ptr == nullptr)
    {
        throw py::error_already_set();
    }
    return comm_ptr;
}

namespace detail
{

std::array<int, 2> mpi_report(MPI_Comm comm)
{
    int size = 0;
    MPI_Comm_size(comm, &size);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    return { rank, size };
}

void export_mpi_bindings(pybind11::module& m, const pybind11::exception<Exception>&
                         /*exception*/)
{
    py::dict features        = m.attr("_named_features");
    features["mpi_bindings"] = 1;

    if (import_mpi4py() < 0)
    {
        throw py::error_already_set();
    }
    m.def(
            "mpi_report",
            [](py::object py_comm) {
                MPI_Comm* comm = get_mpi_comm(py_comm);
                return mpi_report(*comm);
            },
            R"pbdoc(
               Parameters of the MPI context: (rank, size)
        )pbdoc");
}

} // namespace detail
} // end namespace gmxpy
