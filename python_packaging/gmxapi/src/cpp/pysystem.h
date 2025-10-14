/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

#ifndef GMXPY_SYSTEM_H
#define GMXPY_SYSTEM_H

/*! \file
 * \brief Declare helpers for gmxapi::System.
 *
 * \ingroup module_python
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 */

#include <memory>
#include <string>

#include "gmxapi/gmxapi.h"
#include "gmxapi/system.h"

namespace gmxpy
{

std::shared_ptr<gmxapi::System> from_tpr(std::string filename);

class PyContext;

std::shared_ptr<gmxapi::Session> launch(::gmxapi::System* system, PyContext* context);

} // end namespace gmxpy

#endif // header guard
