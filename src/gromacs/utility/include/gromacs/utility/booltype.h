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
 * \brief Define a boolean datatype that can be stored in a std::vector and
 *        have a view on it.
 *
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_utility
 */

#ifndef GMX_BOOLTYPE_H
#define GMX_BOOLTYPE_H

#include <vector>

namespace gmx
{

template<typename>
class ArrayRef;

/*! \brief A clone of a bool as a workaround on the template specialization
 *         of std::vector<bool> that is incompatible with ArrayRef.
 *
 * Use when you need to create an ArrayRef on a vector of boolean values.
 *
 * \note In contrast to bool this type is always initialized to false.
 *
 */
struct BoolType
{
    BoolType() = default;

    /*! \brief Allow implicit construction from plain bool.*/
    BoolType(bool value);

    /*! \brief Conversion to bool. */
    constexpr operator bool() const { return value_; }

    bool value_ = false;
};

/*! \brief
 * Create ArrayRef to bool from reference to std::vector<BoolType>.
 *
 * Allow to easily make views of bool from vectors of BoolType.
 *
 * \see ArrayRef
 */
// NOLINTNEXTLINE(google-runtime-references)
ArrayRef<bool> makeArrayRef(std::vector<BoolType>& boolVector);

/*! \brief
 * Create ArrayRef to const bool from reference to std::vector<BoolType>.
 *
 * Allow to easily make views of const bool from vectors of BoolType.
 *
 * \see ArrayRef
 */
ArrayRef<const bool> makeConstArrayRef(const std::vector<BoolType>& boolVector);

} // namespace gmx
#endif
