/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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

#ifndef GMX_COMPAT_POINTERS_H
#define GMX_COMPAT_POINTERS_H

#include <type_traits>
#include <utility>

#include "gromacs/utility/gmxassert.h"

namespace gmx
{
namespace compat
{

//! Contract-assurance macros that work like a simple version of the GSL ones
//! \{
#define Expects(cond) GMX_ASSERT((cond), "Precondition violation")
#define Ensures(cond) GMX_ASSERT((cond), "Postcondition violation")
//! \}

/*! \libinternal
 * \brief Restricts a pointer or smart pointer to only hold non-null values.
 *
 * Has zero size overhead over T.
 *
 * If T is a pointer (i.e. T == U*) then
 * - allow construction from U*
 * - disallow construction from nullptr_t
 * - disallow default construction
 * - ensure construction from null U* fails (only in debug builds)
 * - allow implicit conversion to U*
 *
 * \todo Eliminate this when we require a version of C++ that supports
 * std::not_null.
 */
template<class T>
class not_null
{
public:
    static_assert(std::is_assignable<T&, std::nullptr_t>::value, "T cannot be assigned nullptr.");

    //! Move constructor. Asserts in debug mode if \c is nullptr.
    template<typename U, typename = std::enable_if_t<std::is_convertible<U, T>::value>>
    constexpr explicit not_null(U&& u) : ptr_(std::forward<U>(u))
    {
        Expects(ptr_ != nullptr);
    }

    //! Simple constructor. Asserts in debug mode if \c u is nullptr.
    template<typename = std::enable_if_t<!std::is_same<std::nullptr_t, T>::value>>
    constexpr explicit not_null(T u) : ptr_(u)
    {
        Expects(ptr_ != nullptr);
    }

    //! Copy constructor.
    template<typename U, typename = std::enable_if_t<std::is_convertible<U, T>::value>>
    constexpr not_null(const not_null<U>& other) : not_null(other.get())
    {
    }

    //! Default constructors and assignment.
    //! \{
    not_null(not_null&& other) noexcept = default;
    not_null(const not_null& other)     = default;
    not_null& operator=(const not_null& other) = default;
    //! \}

    //! Getters
    //! \{
    constexpr T get() const
    {
        Ensures(ptr_ != nullptr);
        return ptr_;
    }

    constexpr   operator T() const { return get(); }
    constexpr T operator->() const { return get(); }
    //! \}

    //! Deleted to prevent compilation when someone attempts to assign a null pointer constant.
    //! \{
    not_null(std::nullptr_t) = delete;
    not_null& operator=(std::nullptr_t) = delete;
    //! \}

    //! Deleted unwanted operators because pointers only point to single objects.
    //! \{
    not_null& operator++()                     = delete;
    not_null& operator--()                     = delete;
    not_null  operator++(int)                  = delete;
    not_null  operator--(int)                  = delete;
    not_null& operator+=(std::ptrdiff_t)       = delete;
    not_null& operator-=(std::ptrdiff_t)       = delete;
    void      operator[](std::ptrdiff_t) const = delete;
    //! \}

private:
    T ptr_;
};

//! Convenience function for making not_null pointers from plain pointers.
template<class T>
not_null<T> make_not_null(T&& t)
{
    return not_null<std::remove_cv_t<std::remove_reference_t<T>>>{ std::forward<T>(t) };
}

//! Convenience function for making not_null pointers from smart pointers.
template<class T>
not_null<typename T::pointer> make_not_null(T& t)
{
    return not_null<typename std::remove_reference_t<T>::pointer>{ t.get() };
}

//! Operators to compare not_null pointers.
//! \{
template<class T, class U>
auto operator==(const not_null<T>& lhs, const not_null<U>& rhs) -> decltype(lhs.get() == rhs.get())
{
    return lhs.get() == rhs.get();
}

template<class T, class U>
auto operator!=(const not_null<T>& lhs, const not_null<U>& rhs) -> decltype(lhs.get() != rhs.get())
{
    return lhs.get() != rhs.get();
}

template<class T, class U>
auto operator<(const not_null<T>& lhs, const not_null<U>& rhs) -> decltype(lhs.get() < rhs.get())
{
    return lhs.get() < rhs.get();
}

template<class T, class U>
auto operator<=(const not_null<T>& lhs, const not_null<U>& rhs) -> decltype(lhs.get() <= rhs.get())
{
    return lhs.get() <= rhs.get();
}

template<class T, class U>
auto operator>(const not_null<T>& lhs, const not_null<U>& rhs) -> decltype(lhs.get() > rhs.get())
{
    return lhs.get() > rhs.get();
}

template<class T, class U>
auto operator>=(const not_null<T>& lhs, const not_null<U>& rhs) -> decltype(lhs.get() >= rhs.get())
{
    return lhs.get() >= rhs.get();
}
//! \}

//! Deleted unwanted arithmetic operators.
//! \{
template<class T, class U>
std::ptrdiff_t operator-(const not_null<T>&, const not_null<U>&) = delete;
template<class T>
not_null<T> operator-(const not_null<T>&, std::ptrdiff_t) = delete;
template<class T>
not_null<T> operator+(const not_null<T>&, std::ptrdiff_t) = delete;
template<class T>
not_null<T> operator+(std::ptrdiff_t, const not_null<T>&) = delete;
//! \}

} // namespace compat
} // namespace gmx

#endif
