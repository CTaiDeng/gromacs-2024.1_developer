/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2011- The GROMACS Authors
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
 * Defines assert macros customized for Gromacs.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inpublicapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_GMXASSERT_H
#define GMX_UTILITY_GMXASSERT_H

#include "current_function.h"

//! \addtogroup module_utility
//! \{

/*! \def GMX_RELEASE_ASSERT
 * \brief
 * Macro for asserts that should also be present in the release version.
 *
 * Regardless of NDEBUG, this macro checks \p condition, and if it is not true,
 * it calls the assert handler.
 *
 * Although this macro currently calls abort() if the assertion fails, it
 * should only be used in a context where it is safe to throw an exception to
 * keep the option open.
 */
#ifdef GMX_DISABLE_ASSERTS
#    define GMX_RELEASE_ASSERT(condition, msg)
#else
#    ifdef _MSC_VER
#        define GMX_RELEASE_ASSERT(condition, msg)                \
            ((void)((condition) ? (void)0                         \
                                : ::gmx::internal::assertHandler( \
                                        #condition, msg, GMX_CURRENT_FUNCTION, __FILE__, __LINE__)))
#    else
// Use an "immediately invoked function expression" to allow being
// used in constexpr context with older GCC versions
// https://akrzemi1.wordpress.com/2017/05/18/asserts-in-constexpr-functions/
#        define GMX_RELEASE_ASSERT(condition, msg)                                                         \
            ((void)((condition) ? (void)0 : [&]() {                                                        \
                ::gmx::internal::assertHandler(#condition, msg, GMX_CURRENT_FUNCTION, __FILE__, __LINE__); \
            }()))
#    endif
#endif
/*! \def GMX_ASSERT
 * \brief
 * Macro for debug asserts.
 *
 * If NDEBUG is defined, this macro expands to nothing.
 * If it is not defined, it will work exactly like ::GMX_RELEASE_ASSERT.
 *
 * \see ::GMX_RELEASE_ASSERT
 */
#ifdef NDEBUG
#    define GMX_ASSERT(condition, msg) ((void)0)
#else
#    define GMX_ASSERT(condition, msg) GMX_RELEASE_ASSERT(condition, msg)
#endif

//! \}

namespace gmx
{

/*! \cond internal */
namespace internal
{

/*! \brief
 * Called when an assert fails.
 *
 * Should not be called directly, but instead through ::GMX_ASSERT or
 * ::GMX_RELEASE_ASSERT.
 *
 * \ingroup module_utility
 */
[[noreturn]] void
assertHandler(const char* condition, const char* msg, const char* func, const char* file, int line);

} // namespace internal
//! \endcond

} // namespace gmx

#endif
