/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * \brief Declarations for Context wrappers.
 *
 * \ingroup module_python
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 */

#ifndef GMXPY_PYCONTEXT_H
#define GMXPY_PYCONTEXT_H

#include "pybind11/pybind11.h"

#include "gmxapi/context.h"
#include "gmxapi/md.h"

#include "gmxpy_exceptions.h"

namespace gmxpy
{

/*!
 * \brief Computing resources are not suitable.
 */
class ResourceError : public ::gmxpy::Exception
{
public:
    using Exception::Exception;
};

using gmxapi::MDArgs;


/*!
 * \brief Wrapper for gmxapi::Context
 *
 * Proxies gmxapi::Context methods and includes additions not yet provided by
 * by upstream library.
 */
class PyContext
{
public:
    PyContext();
    explicit PyContext(std::shared_ptr<gmxapi::Context> context);

    void setMDArgs(const MDArgs& mdArgs);

    std::shared_ptr<gmxapi::Session> launch(const gmxapi::Workflow& work);

    [[nodiscard]] std::shared_ptr<gmxapi::Context> get() const;

    void addMDModule(const pybind11::object& forceProvider) const;

    /*!
     * \brief Borrow shared ownership of the System's container of associated modules.
     *
     * Used with gmxapi::MDHolder to add MD Modules to the simulation to be run.
     *
     * \return handle to be passed to gmxapi::MDHolder
     *
     */
    [[nodiscard]] std::shared_ptr<gmxapi::MDWorkSpec> getSpec() const;

private:
    // TODO(#4467): Directly hold a `gmxapi::Context` object.
    std::shared_ptr<gmxapi::Context>    context_;
    std::shared_ptr<gmxapi::MDWorkSpec> workNodes_;
};

PyContext create_context();
PyContext create_context(pybind11::object communicator);

namespace detail
{

void export_create_context(pybind11::module& module, const pybind11::exception<Exception>& exception);

} // end namespace detail

} // end namespace gmxpy

#endif // GMXPY_PYCONTEXT_H
