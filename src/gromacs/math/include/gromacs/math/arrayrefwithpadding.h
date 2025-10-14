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

/*! \file
 * \brief Declares gmx::ArrayRefWithPadding that refers to memory whose
 * size includes padding for SIMD operations.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \inpublicapi
 * \ingroup module_math
 */
#ifndef GMX_MATH_ARRAYREFWITHPADDING_H
#define GMX_MATH_ARRAYREFWITHPADDING_H

#include <cstddef>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

/*! \brief Interface to a C array of T (or part of a std container of
 * T), that includes padding that is suitable for the kinds of SIMD
 * operations GROMACS uses.
 *
 * High level code e.g. in the force calculation routines need to hold
 * a non-owning view of memory and be able to create ArrayRef objects
 * that view padded or unpadded memory, suitable for the various
 * component routines. This class implements that non-owning view,
 * and the only available functionality refers to its size, and the
 * methods to create such ArrayRef objects.
 *
 * \copydoc ArrayRef
 * \inlibraryapi
 * \ingroup module_math
 */
template<typename T>
class ArrayRefWithPadding
{
public:
    //! Type of values stored in the reference.
    using value_type = T;
    //! Type for representing size of the reference.
    using size_type = Index;
    //! Const pointer to an element.
    using const_pointer = const T*;
    //! Const iterator type to an element.
    using const_iterator = const T*;
    //! Pointer to an element.
    using pointer = T*;
    //! Iterator type to an element.
    using iterator = T*;

    /*! \brief
     * Constructs an empty reference.
     */
    ArrayRefWithPadding() : begin_(nullptr), end_(nullptr), paddedEnd_(nullptr) {}
    /*! \brief
     * Constructs a reference to a particular range.
     *
     * \param[in] begin        Pointer to the beginning of a range.
     * \param[in] end          Pointer to the end of a range without padding.
     * \param[in] paddedEnd    Pointer to the end of a range including padding.
     *
     * Passed pointers must remain valid for the lifetime of this object.
     */
    ArrayRefWithPadding(pointer begin, pointer end, pointer paddedEnd) :
        begin_(begin), end_(end), paddedEnd_(paddedEnd)
    {
        GMX_ASSERT(end >= begin, "Invalid range");
        GMX_ASSERT(paddedEnd >= end, "Invalid range");
    }
    //! Copy constructor
    ArrayRefWithPadding(const ArrayRefWithPadding& o) :
        begin_(o.begin_), end_(o.end_), paddedEnd_(o.paddedEnd_)
    {
    }
    //! Move constructor
    ArrayRefWithPadding(ArrayRefWithPadding&& o) noexcept :
        begin_(std::move(o.begin_)), end_(std::move(o.end_)), paddedEnd_(std::move(o.paddedEnd_))
    {
    }
    /*! \brief Convenience overload constructor to make an ArrayRefWithPadding<const T> from a non-const one.
     *
     * \todo Use std::is_same_v when CUDA 11 is a requirement.
     */
    template<typename U, typename = std::enable_if_t<std::is_same<value_type, const typename std::remove_reference_t<U>::value_type>::value>>
    ArrayRefWithPadding(U&& o)
    {
        auto constArrayRefWithPadding = o.constArrayRefWithPadding();
        this->swap(constArrayRefWithPadding);
    }
    //! Copy assignment operator
    ArrayRefWithPadding& operator=(ArrayRefWithPadding const& o)
    {
        if (&o != this)
        {
            begin_     = o.begin_;
            end_       = o.end_;
            paddedEnd_ = o.paddedEnd_;
        }
        return *this;
    }
    //! Move assignment operator
    ArrayRefWithPadding& operator=(ArrayRefWithPadding&& o) noexcept
    {
        if (&o != this)
        {
            begin_     = std::move(o.begin_);
            end_       = std::move(o.end_);
            paddedEnd_ = std::move(o.paddedEnd_);
        }
        return *this;
    }

    //! Return the size of the view, i.e with the padding.
    size_type size() const { return paddedEnd_ - begin_; }
    //! Whether the reference refers to no memory.
    bool empty() const { return begin_ == end_; }

    //! Returns an ArrayRef of elements that does not include the padding region.
    ArrayRef<T> unpaddedArrayRef() { return { begin_, end_ }; }
    //! Returns an ArrayRef of const elements that does not include the padding region.
    ArrayRef<const T> unpaddedConstArrayRef() const { return { begin_, end_ }; }
    //! Returns an ArrayRef of elements that does include the padding region.
    ArrayRef<T> paddedArrayRef() { return { begin_, paddedEnd_ }; }
    //! Returns an ArrayRef of const elements that does include the padding region.
    ArrayRef<const T> paddedConstArrayRef() const { return { begin_, paddedEnd_ }; }
    //! Returns an identical ArrayRefWithPadding that refers to const elements.
    ArrayRefWithPadding<const T> constArrayRefWithPadding() const
    {
        return { begin_, end_, paddedEnd_ };
    }
    /*! \brief
     * Swaps referenced memory with the other object.
     *
     * The actual memory areas are not modified, only the references are
     * swapped.
     */
    void swap(ArrayRefWithPadding<T>& other)
    {
        std::swap(begin_, other.begin_);
        std::swap(end_, other.end_);
        std::swap(paddedEnd_, other.paddedEnd_);
    }

private:
    pointer begin_;
    pointer end_;
    pointer paddedEnd_;
};

} // namespace gmx

#endif
