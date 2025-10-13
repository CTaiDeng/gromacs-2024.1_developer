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
 * \brief Exception translation for exceptions propagating from dependencies.
 *
 * Note that C++ exception objects are not actually bound to Python objects
 * because C++ cannot give ownership of a thrown exception to the interpreter.
 * Instead, we catch C++ exceptions and translate them into Python exceptions.
 *
 * As of pybind11 2.2, pybind does not provide a way to automatically express
 * C++ exception class inheritance to Python, and only supports simple Python
 * exceptions initialized with strings at the time of translation. (Automatic
 * translation uses `std::exception::what()`)
 *
 * Currently, we restrict ourselves to this simple translation. Future versions
 * may use the Python C API directly to support richer exception classes.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 *
 * \ingroup module_python
 */

#ifndef GMXPY_WRAPPED_EXCEPTIONS_H
#define GMXPY_WRAPPED_EXCEPTIONS_H

#include <exception>

#include "pybind11/pybind11.h"

#include "gmxpy_exceptions.h"

namespace gmxpy::detail
{

/*! \brief Register Python exceptions for translated external C++ exceptions.
 *
 * \param m Python extension module in which to register exceptions.
 * \param baseException Base from which exceptions in *m* should derive.
 *
 * Wrap known gmxapi::Exception subclasses as Python exceptions derived from
 * :py:class:`gmxapi.exceptions.Error`.
 *
 * The exact Python exceptions defined depends on the supporting GROMACS libraries
 * available at build time. (Implementation is determined by CMake when the build
 * system is configured.)
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 *
 * \ingroup module_python
 */
void export_wrapped_exceptions(pybind11::module& m, const pybind11::exception<Exception>& baseException);

} // namespace gmxpy::detail
#endif // GMXPY_WRAPPED_EXCEPTIONS_H
