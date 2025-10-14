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

/*! \internal \file
 *
 * \brief Implements routines in booltype.h .
 *
 * \author Christian Blau <cblau.mail@gmail.com>
 */

#include "gmxpre.h"

#include "gromacs/utility/booltype.h"

#include "gromacs/utility/arrayref.h"

namespace gmx
{

BoolType::BoolType(bool value) : value_{ value } {}

ArrayRef<bool> makeArrayRef(std::vector<BoolType>& boolVector)
{
    return { reinterpret_cast<bool*>(boolVector.data()),
             reinterpret_cast<bool*>(boolVector.data() + boolVector.size()) };
}

ArrayRef<const bool> makeConstArrayRef(const std::vector<BoolType>& boolVector)
{
    return { reinterpret_cast<const bool*>(boolVector.data()),
             reinterpret_cast<const bool*>(boolVector.data() + boolVector.size()) };
}


} // namespace gmx
