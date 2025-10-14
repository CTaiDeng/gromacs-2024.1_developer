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


#ifndef GMXPY_MPI_BINDINGS_H
#define GMXPY_MPI_BINDINGS_H

#include <mpi.h>

#include "pybind11/pybind11.h"

#include "gmxapi/context.h"

namespace gmxpy
{

// Base exception forward declared for gmxpy_exceptions.h.
class Exception;

/*!
 * \brief Get a pointer to the MPI_Comm wrapped in an mpi4py Comm
 * \param communicator (:py:class:`mpi4py.MPI.Comm`): wrapped MPI communicator.
 * \return Pointer to C object.
 */
MPI_Comm* get_mpi_comm(pybind11::object communicator);

/*!
 * \brief Adapter to the gmxapi offer_comm protocol.
 * \param communicator (:py:class:`mpi4py.MPI.Comm`): wrapped MPI communicator.
 * \return gmxapi C++ Context handle.
 *
 * Implementation is selected by CMake, depending on available GROMACS library support.
 * See :file:`gmxapi/mpi/gmxapi_mpi.h` and the :cpp:func:`gmxapi::assignResource()` template.
 *
 * \throws if communicator is invalid (e.g. MPI_COMM_NULL) or not usable (e.g. too big)
 */
gmxapi::Context context_from_py_comm(pybind11::object communicator);

namespace detail
{

/*!
 * \brief Register bindings for MPI and MPI-enabled GROMACS, if possible.
 * \param m (:cpp:class:`pybind11::module`): The Python module that is in the process of being
 * imported. \param exception Module base exception from which additional exceptions should derive.
 */
void export_mpi_bindings(pybind11::module& m, const pybind11::exception<Exception>& exception);

} // end namespace detail

} // end namespace gmxpy

#endif // GMXPY_MPI_BINDINGS_H
