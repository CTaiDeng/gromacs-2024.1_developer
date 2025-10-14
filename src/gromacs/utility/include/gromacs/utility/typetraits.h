/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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

/*! \libinternal \file
 * \brief
 * Declares type traits
 *
 * \inlibraryapi
 * \ingroup module_utility
 */

#ifndef GMX_UTILITY_TYPETRAITS_H
#define GMX_UTILITY_TYPETRAITS_H

#include <type_traits>

namespace gmx
{

/*! \libinternal \brief
 * Is true if type is a std::integral_constant
 *
 * If the optional integral type is given, than it is only true for a
 * std::integral_constant of that integral type.
 *
 * \tparam T type to check
 * \tparam Int optional integral type
 */
template<typename T, typename Int = void>
struct isIntegralConstant : public std::false_type
{
};

template<typename Int, Int N>
struct isIntegralConstant<std::integral_constant<Int, N>, void> : public std::true_type
{
};

template<typename Int, Int N>
struct isIntegralConstant<std::integral_constant<Int, N>, Int> : public std::true_type
{
};

} // namespace gmx

#endif
