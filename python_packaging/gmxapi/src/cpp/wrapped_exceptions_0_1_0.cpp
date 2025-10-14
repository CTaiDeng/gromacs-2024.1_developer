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
 * \brief Exception translation for gmxapi library < 0.3.1.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 *
 * \ingroup module_python
 */
#include "wrapped_exceptions.h"

#include "gmxapi/exceptions.h"
#include "gmxapi/version.h"

#include "gmxpy_exceptions.h"


namespace gmxpy::detail
{

namespace py = pybind11;


void export_wrapped_exceptions(pybind11::module& m, const pybind11::exception<Exception>& baseException)
{
    assert(!gmxapi::Version::isAtLeast(0, 3, 1) && "CMake found the wrong source file.");

    // Map gmxapi exceptions from gmxapi/exceptions.h to Python package exceptions.
    // Note: C++ exception translation occurs in revers order of registration,
    // So derived exceptions must be registered after their base exceptions.
    // TODO: We could have more informative exception class docstrings
    //   by linking to online docs or if we had a way to reuse doxygen docs.

    {
        auto exception =
                py::register_exception<gmxapi::ProtocolError>(m, "ProtocolError", baseException.ptr());
        exception.doc() = "Behavioral protocol violated.";
    }

    {
        const auto* const missing_implementation_doc =
                "Expected feature is not implemented.\n\n"
                ".. deprecated:: 0.3\n"
                "    Use MissingImplementationError instead.\n";
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        auto exception = py::register_exception<gmxapi::NotImplementedError>(
                m, "NotImplementedError", baseException.ptr());
#pragma clang diagnostic pop
        exception.doc() = missing_implementation_doc;
    }

    {
        auto exception =
                py::register_exception<gmxapi::UsageError>(m, "UsageError", baseException.ptr());
        exception.doc() = "Unacceptable API usage.";
    }
}


} // end namespace gmxpy::detail
