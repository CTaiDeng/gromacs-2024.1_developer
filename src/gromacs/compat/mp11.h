/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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
 * \brief Provides ported functions/classes from boost::mp11
 *
 * Adapted from the Boost Library 1.67
 *
 * \author Roland Schulz <roland.schulz@intel.com>
 * \ingroup module_compat
 * \inlibraryapi
 */
#ifndef GMX_COMPAT_MP11_H
#define GMX_COMPAT_MP11_H

#include <utility>

#include "gromacs/utility/exceptions.h"

namespace gmx
{
namespace compat
{

/** \internal \brief Simplified analogue of boost::mp11::mp_with_index, compatible only with C++17 and up.
 *
 * \c mp_with_index<N>(i, f) calls \p f with \c mp_size_t<i>() and returns the result.
 * \p i must be less than \p N.
 *
 * Example usage:
 * \code
    constexpr int foo_max = 3;
    template<int i, typename = std::enable_if_t<(i < foo_max)>>
    bool constexpr foo();

    bool bar(int i)
    {
        return mp_with_index<foo_max>(i, [](auto i) {
            return foo<i>();
        });
    }
 * \endcode
 */
template<std::size_t N, class F, typename std::enable_if<(N <= 1)>::type* = nullptr>
static auto mp_with_index(std::size_t i, F&& f)
{
    // Last step of recursion. Must have one active "return" for proper type deduction.
    if (i == N - 1)
    {
        return std::forward<F>(f)(std::integral_constant<std::size_t, N - 1>());
    }
    else
    {
        const std::string errorMessage =
                "Invalid arguments of mp_with_index (i=" + std::to_string(i) + ")";
        GMX_THROW(InternalError(errorMessage));
    }
}

// Doxygen does not like recursive templates.
//! \cond
template<std::size_t N, class F, typename std::enable_if<(N > 1)>::type* = nullptr>
static auto mp_with_index(std::size_t i, F&& f)
{
    if (i == N - 1)
    {
        return std::forward<F>(f)(std::integral_constant<std::size_t, N - 1>());
    }
    else
    {
        return mp_with_index<N - 1>(i, std::forward<F>(f));
    }
}
//! \endcond

} // namespace compat
} // namespace gmx

#endif
