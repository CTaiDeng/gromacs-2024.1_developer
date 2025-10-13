/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

#ifndef GMXAPI_SYSTEM_IMPL_H
#define GMXAPI_SYSTEM_IMPL_H

/*! \file
 * \brief Declare implementation details for gmxapi::System.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 * \ingroup gmxapi
 */

#include <string>

#include "gmxapi/system.h"

namespace gmxapi
{

class Context;
class Workflow;

/*!
 * \brief Private implementation for gmxapi::System
 *
 * \ingroup gmxapi
 */
class System::Impl final
{
public:
    /*! \cond */
    ~Impl();

    Impl(Impl&& /*unused*/) noexcept;
    Impl& operator=(Impl&& source) noexcept;
    /*! \endcond */

    /*!
     * \brief Initialize from a work description.
     *
     * \param workflow Simulation work to perform.
     */
    explicit Impl(std::unique_ptr<gmxapi::Workflow> workflow) noexcept;

    /*!
     * \brief Launch the configured simulation.
     *
     * \param context Runtime execution context in which to run simulation.
     * \return Ownership of a new simulation session.
     *
     * The session is returned as a shared pointer so that the Context can
     * maintain a weak reference to it via std::weak_ptr.
     */
    std::shared_ptr<Session> launch(const std::shared_ptr<Context>& context);

    //! Description of simulation work.
    std::shared_ptr<Workflow> workflow_;

    /*!
     * \brief Specified simulation work.
     *
     * \todo merge Workflow and MDWorkSpec
     */
    std::shared_ptr<gmxapi::MDWorkSpec> spec_;
};

} // end namespace gmxapi

#endif // header guard
