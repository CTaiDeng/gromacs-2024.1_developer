/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * \brief
 * Declares GMX_CURRENT_FUNCTION for getting the current function name.
 *
 * The implementation is essentially copied from Boost 1.55 (see reference below).
 *
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_CURRENT_FUNCTION_H
#define GMX_UTILITY_CURRENT_FUNCTION_H

/*! \def GMX_CURRENT_FUNCTION
 * \brief
 * Expands to a string that provides the name of the current function.
 *
 * \ingroup module_utility
 */

//
//  boost/current_function.hpp - BOOST_CURRENT_FUNCTION
//
//  Copyright (c) 2002 Peter Dimov and Multi Media Ltd.
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
//  http://www.boost.org/libs/utility/current_function.html
//

namespace gmx
{

namespace internal
{

//! Helper for defining GMX_CURRENT_FUNCTION.
inline void current_function_helper()
{

#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) \
        || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)

#    define GMX_CURRENT_FUNCTION __PRETTY_FUNCTION__

#elif defined(__DMC__) && (__DMC__ >= 0x810)

#    define GMX_CURRENT_FUNCTION __PRETTY_FUNCTION__

#elif defined(__FUNCSIG__)

#    define GMX_CURRENT_FUNCTION __FUNCSIG__

#elif (defined(__IBMCPP__) && (__IBMCPP__ >= 500))

#    define GMX_CURRENT_FUNCTION __FUNCTION__

#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)

#    define GMX_CURRENT_FUNCTION __FUNC__

#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)

#    define GMX_CURRENT_FUNCTION __func__

#else

#    define GMX_CURRENT_FUNCTION "(unknown)"

#endif
}

} // namespace internal

} // namespace gmx

#endif
