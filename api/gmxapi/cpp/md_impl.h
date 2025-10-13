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

#ifndef GMXAPI_MD_IMPL_H
#define GMXAPI_MD_IMPL_H
/*! \file
 * \brief Declarations for molecular dynamics API implementation details.
 *
 * \ingroup gmxapi
 */

#include <memory>

#include "gmxapi/gmxapi.h"
#include "gmxapi/md.h"

namespace gmxapi
{

class MDWorkSpec;

/*!
 * \brief Implementation class to hide guts of MDHolder
 *
 * Holds the gmxapi interface for an object that can help instantiate the gmx::MdRunner
 */
class MDHolder::Impl
{
public:
    /*!
     * \brief Construct by capturing a messaging object.
     *
     * \param spec operations specified for a workflow and the means to instantiate them.
     */
    explicit Impl(std::shared_ptr<MDWorkSpec>&& spec);

    /*!
     * \brief Shared ownership of the gmxapi object used for higher level message passing.
     */
    std::shared_ptr<MDWorkSpec> spec_{ nullptr };
};

} // namespace gmxapi

#endif // header guard
