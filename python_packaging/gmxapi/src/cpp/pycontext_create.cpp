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
 * \brief Create a PyContext for gmxapi >= 0.2.0
 *
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 *
 * \ingroup module_python
 */


#include "gmxpy_exceptions.h"
#include "mpi_bindings.h"
#include "pycontext.h"

namespace py = pybind11;

namespace gmxpy
{

PyContext create_context()
{
    auto context     = gmxapi::createContext();
    auto context_ptr = std::make_shared<gmxapi::Context>(std::move(context));
    return PyContext(std::move(context_ptr));
}

PyContext create_context(py::object communicator)
{
    auto context     = context_from_py_comm(communicator);
    auto context_ptr = std::make_shared<gmxapi::Context>(std::move(context));
    return PyContext(std::move(context_ptr));
}

namespace detail
{

void export_create_context(pybind11::module& m, const pybind11::exception<Exception>& exception)
{
    py::dict features          = m.attr("_named_features");
    features["create_context"] = 1;

    export_mpi_bindings(m, exception);

    m.def(
            "create_context",
            []() { return create_context(); },
            "Initialize a new API Context to manage resources and software environment.");
    m.def(
            "create_context",
            [](const py::object& resource) { return create_context(resource); },
            "Initialize a new API Context to manage resources and software environment.");
}
} // end namespace detail

} // namespace gmxpy
