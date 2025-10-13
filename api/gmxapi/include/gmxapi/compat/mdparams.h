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
 * \brief Compatibility header for simulation parameters.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 * \ingroup gmxapi_compat
 */

#ifndef GMXAPICOMPAT_MDPARAMS_H
#define GMXAPICOMPAT_MDPARAMS_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gmxapi/gmxapi.h"
#include "gmxapi/gmxapicompat.h"

namespace gmxapicompat
{

// Forward declaration for private implementation class for GmxMdParams
class GmxMdParamsImpl;

class GmxMdParams
{
public:
    GmxMdParams();
    ~GmxMdParams();
    GmxMdParams(const GmxMdParams&) = delete;
    GmxMdParams& operator=(const GmxMdParams&) = delete;
    GmxMdParams(GmxMdParams&& /*unused*/) noexcept;
    GmxMdParams& operator=(GmxMdParams&& /*unused*/) noexcept;

    explicit GmxMdParams(std::unique_ptr<GmxMdParamsImpl>&& impl);

    std::unique_ptr<GmxMdParamsImpl> params_;
};

/*!
 * \brief Get the list of parameter key words.
 *
 * \param params molecular simulation parameters object reference.
 * \return A new vector of parameter names.
 *
 * \note The returned data is a copy. Modifying the return value has no affect on
 * the original object inspected.
 */
std::vector<std::string> keys(const GmxMdParams& params);

} // end namespace gmxapicompat

#endif // GMXAPICOMPAT_MDPARAMS_H
