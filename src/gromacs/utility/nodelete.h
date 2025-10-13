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

/*! \libinternal \file
 * \brief
 * Declares no_delete deleter for std::shared_ptr.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_NODELETE_H
#define GMX_UTILITY_NODELETE_H

namespace gmx
{

/*! \libinternal \brief
 * Deleter for std::shared_ptr that does nothing.
 *
 * This is useful for cases where a class needs to keep a reference to another
 * class, and optionally also manage the lifetime of that other class.
 * The simplest construct (that does not force all callers to use heap
 * allocation and std::shared_ptr for the referenced class) is to use a
 * single std::shared_ptr to hold that reference, and use no_delete as the
 * deleter if the lifetime is managed externally.
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
template<class T>
struct no_delete
{
    //! Deleter that does nothing.
    void operator()(T* /*unused*/) {}
};

} // namespace gmx

#endif
