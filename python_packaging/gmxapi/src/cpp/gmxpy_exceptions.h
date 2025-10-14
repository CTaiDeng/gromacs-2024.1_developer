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
 * \brief C++ and Python exceptions throwable through this library.
 *
 * Establish the C++ exception hierarchy for the ::gmxpy namespace.
 *
 * Provide an export function to register Python exceptions at module import.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 *
 * \ingroup module_python
 */

#ifndef GMXPY_EXCEPTIONS_H
#define GMXPY_EXCEPTIONS_H

#include <stdexcept>

#include "pybind11/pybind11.h"

namespace gmxpy
{

/*! \brief Base exception for gmxapi Python compiled extension module.
 *
 * Exceptions thrown in the gmxpy namespace are descended from gmxpy::Exception
 * or there is a bug.
 */
class Exception : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/*! \brief An API feature is not available in the current installation.
 *
 * This may occur when a new gmxapi Python package is installed with an older
 * GROMACS installation that does not have the library support for a newer
 * feature.
 */
class FeatureNotAvailable : public Exception
{
public:
    //! \cond
    // Constructor definitions are hidden to allow for extra behavior to be
    // added in the future. For instance, it would be nice to add the library
    // versions to the error message for easy reference.
    FeatureNotAvailable();
    ~FeatureNotAvailable() override;
    FeatureNotAvailable(const FeatureNotAvailable& /*unused*/);
    FeatureNotAvailable& operator=(const FeatureNotAvailable& /*unused*/);

    FeatureNotAvailable(FeatureNotAvailable&& /*unused*/) noexcept;
    FeatureNotAvailable& operator=(FeatureNotAvailable&& /*unused*/) noexcept;
    //! \endcond

    /*!
     * \brief Describe missing library support for an API call.
     *
     * \param what_arg Message string for the exception.
     *
     * The message *should* indicate a corresponding named feature
     * that is detectable with :py:func:`gmxapi._gmxapi.has_feature`.
     * @{
     */
    explicit FeatureNotAvailable(const std::string& what_arg);
    explicit FeatureNotAvailable(const char* what_arg);
    // @}
};


namespace detail
{

/*!
 * \brief Register Python exceptions throwable from this C++ extension module.
 *
 * Registers Python exceptions for ::gmxpy C++ exceptions (exceptions
 * originating in this module), as well as handlers for C++ exceptions
 * propagated from dependencies. Provides additional basic wrapping for unknown
 * exceptions.
 *
 * Acquires a reference to the Python :py:class:`gmxapi.exceptions.Error` and
 * derives a base Python exception for this extension module.
 *
 * This export function should be called as early as possible on module import
 * since other bindings and export functions may depend on the Python exceptions.
 *
 * \param m Python extension module in which to define the Exceptions.
 *
 * \return Reference to a static object that can be the base exception for
 * Exceptions defined in sub-components (exported with other `export_X` functions
 * in `module.cpp`).
 */
const pybind11::exception<Exception>& export_exceptions(pybind11::module& m);

} // namespace detail
} // namespace gmxpy
#endif // GMXPY_EXCEPTIONS_H
