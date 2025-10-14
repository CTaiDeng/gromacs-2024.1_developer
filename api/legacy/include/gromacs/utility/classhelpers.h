/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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

/*! \file
 * \brief
 * Declares common utility classes and macros.
 *
 * This header contains helpers used to implement classes in the library.
 * It is installed, because the helpers are used in installed headers, but
 * typically users of the library should not need to be aware of these helpers.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_CLASSHELPERS_H
#define GMX_UTILITY_CLASSHELPERS_H

namespace gmx
{

#ifdef DOXYGEN
/*! \brief
 * Macro to declare a class non-copyable and non-assignable.
 *
 * For consistency, should appear last in the class declaration.
 *
 * \ingroup module_utility
 */
#    define GMX_DISALLOW_COPY_AND_ASSIGN(ClassName)
#else
#    define GMX_DISALLOW_COPY_AND_ASSIGN(ClassName)      \
        ClassName& operator=(const ClassName&) = delete; \
        ClassName(const ClassName&)            = delete
#endif
#ifdef DOXYGEN
/*! \brief
 * Macro to declare a class non-copyable, non-movable, non-copy-assignable and
 * non-move-assignable.
 *
 * For consistency, should appear last in the class declaration.
 *
 * \ingroup module_utility
 */
#    define GMX_DISALLOW_COPY_MOVE_AND_ASSIGN(ClassName)
#else
#    define GMX_DISALLOW_COPY_MOVE_AND_ASSIGN(ClassName)                                                            \
        ClassName& operator=(const ClassName&) = delete;                                                            \
        ClassName(const ClassName&)            = delete;                                                            \
        ClassName& operator=(ClassName&&) = delete; /* NOLINT(misc-macro-parentheses,bugprone-macro-parentheses) */ \
        ClassName(ClassName&&) = delete /* NOLINT(misc-macro-parentheses,bugprone-macro-parentheses) */
#endif
/*! \brief
 * Macro to declare a class non-assignable.
 *
 * For consistency, should appear last in the class declaration.
 *
 * \ingroup module_utility
 */
#define GMX_DISALLOW_ASSIGN(ClassName) ClassName& operator=(const ClassName&) = delete

// clang-format off
#ifdef DOXYGEN
/*! \brief
 * Macro to declare default constructors
 *
 * Intended for copyable interfaces or bases classes which require to create custom
 * destructor (e.g. protected or virtual) but need the default constructors.
 *
 * \ingroup module_utility
 */
#    define GMX_DEFAULT_CONSTRUCTORS(ClassName)
#else
#    define GMX_DEFAULT_CONSTRUCTORS(ClassName)                                                                           \
        ClassName() = default;                                                                                            \
        ClassName& operator=(const ClassName&) = default; /* NOLINT(misc-macro-parentheses,bugprone-macro-parentheses) */ \
        ClassName(const ClassName&) = default;                                                                            \
        ClassName& operator=(ClassName&&) = default; /* NOLINT(misc-macro-parentheses,bugprone-macro-parentheses) */      \
        ClassName(ClassName&&) = default /* NOLINT(misc-macro-parentheses,bugprone-macro-parentheses) */
#endif
//clang-format on

} // namespace gmx

#endif
