/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * \brief Declares an allocator that can use default initialization instead
 * of values initialization. This is useful for improving performance of
 * resize() in standard vectors for buffers in performance critical code.
 *
 * \author Berk Hess <hess@kth.se>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_DEFAULTINITIALIZATIONALLOCATOR_H
#define GMX_UTILITY_DEFAULTINITIALIZATIONALLOCATOR_H

#include <memory>

namespace gmx
{

/*! \libinternal \brief Allocator adaptor that interposes construct() calls to
 * convert value initialization into default initialization.
 *
 * This can be used to avoid initialization e.g. on resize() in std::vector.
 */
template<typename T, typename A = std::allocator<T>>
class DefaultInitializationAllocator : public A
{
    typedef std::allocator_traits<A> a_t;

public:
    template<typename U>
    struct rebind
    {
        using other = DefaultInitializationAllocator<U, typename a_t::template rebind_alloc<U>>;
    };

    using A::A;

    /*! \brief Constructs an object and default initializes
     *
     * \todo Use std::is_nothrow_default_constructible_v when CUDA 11 is a requirement.
     */
    template<typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value)
    {
        ::new (static_cast<void*>(ptr)) U;
    }

    /*! \brief Constructs an object and value initializes */
    template<typename U, typename... Args>
    void construct(U* ptr, Args&&... args)
    {
        a_t::construct(static_cast<A&>(*this), ptr, std::forward<Args>(args)...);
    }
};

} // namespace gmx

#endif // GMX_UTILITY_DEFAULTINITIALIZATIONALLOCATOR_H
