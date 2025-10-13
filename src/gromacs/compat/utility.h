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

/*! \libinternal \file
 * \brief Provides backported functions/classes from utility
 *
 * \todo Remove when CUDA 11 is a requirement.
 *
 * \author Roland Schulz <roland.schulz@intel.com>
 * \ingroup module_compat
 * \inlibraryapi
 */
#ifndef GMX_COMPAT_UTILITY_H
#define GMX_COMPAT_UTILITY_H
namespace gmx
{
namespace compat
{
//! Forms lvalue reference to const type of t
template<class T>
constexpr const T& as_const(T& t) noexcept
{
    return t;
}
} // namespace compat
} // namespace gmx
#endif
