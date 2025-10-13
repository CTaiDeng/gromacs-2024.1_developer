/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

/*! \inpublicapi \file
 * \brief
 * Implements nblib exception class
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#ifndef NBLIB_EXCEPTION_H
#define NBLIB_EXCEPTION_H

#include <exception>
#include <string>

namespace nblib
{

/*! \brief Base nblib exception class
 *
 * All nblib exceptions derive from this class and simply forward their message. This allows
 * exceptions to be handled uniformly across different exception types.
 */
class NbLibException : public std::exception
{
public:
    [[maybe_unused]] explicit NbLibException(const std::string& message) :
        message_("NbLib Exception: " + message)
    {
    }

    //! Overrides the what() in std::exception
    [[nodiscard]] const char* what() const noexcept override { return message_.c_str(); }

    //! Convenience call in case a string is wanted instead of a const char*
    [[nodiscard]] const std::string& reason() const& { return message_; }

private:
    std::string message_;
};

/*! \brief The exception type for user input errors
 *
 * The message should give users some hint as to how to remedy the error.
 */
class InputException final : public NbLibException
{
public:
    using NbLibException::NbLibException;
};

} // namespace nblib
#endif // NBLIB_EXCEPTION_H
