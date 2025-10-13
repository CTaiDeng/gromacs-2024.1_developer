/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * \brief Declares allocation policy classes and allocators that are
 * used to make library containers compatible with alignment
 * requirements of particular hardware, e.g. memory operations for
 * SIMD or accelerators.
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inpublicapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_ALIGNEDALLOCATOR_H
#define GMX_UTILITY_ALIGNEDALLOCATOR_H

#include <cstddef>

#include "gromacs/utility/allocator.h"

namespace gmx
{

/*! \libinternal \brief Policy class for configuring gmx::Allocator, to manage
 * allocations of aligned memory for SIMD code.
 */
class AlignedAllocationPolicy
{
public:
    /*! \brief Return the alignment size. */
    static std::size_t alignment();
    /*! \brief Allocate memory aligned to alignment() bytes.
     *
     *  \param bytes Amount of memory (bytes) to allocate. It is valid to ask for
     *               0 bytes, which will return a non-null pointer that is properly
     *               aligned and padded (but that you should not use).
     *
     * \return Valid pointer if the allocation worked, otherwise nullptr.
     *
     * The memory will always be aligned to 128 bytes, which is our
     * estimate of the longest cache lines on architectures currently in use.
     * It will also be padded by the same amount at the end of the
     * area, to help avoid false cache sharing.
     *
     *  \note Memory allocated with this routine must be released with
     *        gmx::AlignedAllocationPolicy::free(), and absolutely not the system free().
     */
    static void* malloc(std::size_t bytes);
    /*! \brief Free aligned memory
     *
     *  \param p  Memory pointer previously returned from malloc()
     *
     *  \note This routine should only be called with pointers obtained from
     *        gmx::AlignedAllocationPolicy::malloc(), and absolutely not any
     *        pointers obtained the system malloc().
     */
    static void free(void* p);
};

/*! \brief Aligned memory allocator.
 *
 *  \tparam T          Type of objects to allocate
 *
 * This convenience partial specialization can be used for the
 * optional allocator template parameter in standard library
 * containers, which is necessary e.g. to use SIMD aligned load and
 * store operations on data in those containers. The memory will
 * always be aligned according to the behavior of
 * AlignedAllocationPolicy.
 */
template<class T>
using AlignedAllocator = Allocator<T, AlignedAllocationPolicy>;


/*! \brief Return the memory page size on this system
 *
 * Implements the "construct on first use" idiom to avoid the static
 * initialization order fiasco where a possible static page-aligned
 * container would be initialized before the alignment variable was.
 *
 * Note that thread-safety is guaranteed by the C++11 language
 * standard. */
std::size_t pageSize();

/*! \libinternal \brief Policy class for configuring gmx::Allocator,
 * to manage allocations of page-aligned memory that can be locked for
 * asynchronous transfer to GPU devices.
 */
class PageAlignedAllocationPolicy
{
public:
    /*! \brief Return the alignment size of memory pages on this system.
     *
     * Queries sysconf/WinAPI, otherwise guesses 4096. */
    static std::size_t alignment();
    /*! \brief Allocate memory aligned to alignment() bytes.
     *
     *  \param bytes Amount of memory (bytes) to allocate. It is valid to ask for
     *               0 bytes, which will return a non-null pointer that is properly
     *               aligned and padded (but that you should not use).
     *
     * \return Valid pointer if the allocation worked, otherwise nullptr.
     *
     *  \note Memory allocated with this routine must be released with
     *        gmx::PageAlignedAllocationPolicy::free(), and absolutely not the system free().
     */
    static void* malloc(std::size_t bytes);
    /*! \brief Free aligned memory
     *
     *  \param p  Memory pointer previously returned from malloc()
     *
     *  \note This routine should only be called with pointers obtained from
     *        gmx::PageAlignedAllocationPolicy::malloc(), and absolutely not any
     *        pointers obtained the system malloc().
     */
    static void free(void* p);
};

/*! \brief PageAligned memory allocator.
 *
 *  \tparam T          Type of objects to allocate
 *
 * This convenience partial specialization can be used for the
 * optional allocator template parameter in standard library
 * containers, which is necessary for locking memory pages for
 * asynchronous transfer between a GPU device and the host.  The
 * memory will always be aligned according to the behavior of
 * PageAlignedAllocationPolicy.
 */
template<class T>
using PageAlignedAllocator = Allocator<T, PageAlignedAllocationPolicy>;

} // namespace gmx

#endif // GMX_UTILITY_ALIGNEDALLOCATOR_H
