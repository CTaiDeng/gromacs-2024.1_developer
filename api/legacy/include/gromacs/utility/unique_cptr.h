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
 * Declares gmx::unique_cptr and gmx::sfree_guard.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_UNIQUE_PTR_SFREE_H
#define GMX_UTILITY_UNIQUE_PTR_SFREE_H

#include <cstdlib>

#include <memory>

#include "gromacs/utility/smalloc.h"

namespace gmx
{

/*! \brief Wrapper of standard library free(), to be used as
 * unique_cptr deleter for memory allocated by malloc, e.g. by an
 * external library such as TNG. */
template<class T>
inline void free_wrapper(T* p)
{
    free(p);
}

//! sfree wrapper to be used as unique_cptr deleter
template<class T>
inline void sfree_wrapper(T* p)
{
    sfree(p);
}

//! \internal \brief wrap function into functor to be used as deleter
template<class T, void D(T*)>
struct functor_wrapper
{
    //! call wrapped function
    void operator()(T* t) { D(t); }
};

//! unique_ptr which takes function pointer (has to return void) as template argument
template<typename T, void D(T*) = sfree_wrapper>
using unique_cptr = std::unique_ptr<T, functor_wrapper<T, D>>;

//! Simple guard which calls sfree. See unique_cptr for details.
typedef unique_cptr<void> sfree_guard;


//! Create unique_ptr with any deleter function or lambda
template<typename T, typename D>
std::unique_ptr<T, D> create_unique_with_deleter(T* t, D d)
{
    return std::unique_ptr<T, D>(t, d);
}

} // namespace gmx

#endif
