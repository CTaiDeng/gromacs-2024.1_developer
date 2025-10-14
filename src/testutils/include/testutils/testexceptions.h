/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2011- The GROMACS Authors
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

/*! \libinternal \file
 * \brief
 * Exception classes for errors in tests.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_testutils
 */
#ifndef GMX_TESTUTILS_TESTEXCEPTIONS_H
#define GMX_TESTUTILS_TESTEXCEPTIONS_H

#include <string>

#include "gromacs/utility/exceptions.h"

namespace gmx
{
namespace test
{

/*! \libinternal \brief
 * Exception class for reporting errors in tests.
 *
 * This exception should be used for error conditions that are internal to the
 * test, i.e., do not indicate errors in the tested code.
 *
 * \ingroup module_testutils
 */
class TestException : public GromacsException
{
public:
    /*! \brief
     * Creates a test exception object with the provided detailed reason.
     *
     * \param[in] reason Detailed reason for the exception.
     */
    explicit TestException(const std::string& reason) : GromacsException(reason) {}
    /*! \brief
     * Creates a test exception based on another GromacsException object.
     *
     * \param[in] base  Exception to wrap.
     *
     * \see GMX_THROW_WRAPPER_TESTEXCEPTION
     */
    explicit TestException(const GromacsException& base) : GromacsException(base) {}

    int errorCode() const override { return -1; }
};

/*! \brief
 * Macro for throwing a TestException that wraps another exception.
 *
 * \param[in] e    Exception object to wrap.
 *
 * This macro is intended for wrapping exceptions thrown by Gromacs methods
 * that are called from a test for the test's internal purposes.  It wraps the
 * exception in a TestException to make it possible to tell from the type of
 * the exception whether the exception was thrown by the code under test, or by
 * the test code itself.
 *
 * \p e should evaluate to an instance of an object derived from
 * GromacsException.
 *
 * Typical usage in test code:
 * \code
   try
   {
       // some code that may throw a GromacsException
   }
   catch (const GromacsException &ex)
   {
       GMX_THROW_WRAPPER_TESTEXCEPTION(ex);
   }
 * \endcode
 */
#define GMX_THROW_WRAPPER_TESTEXCEPTION(e) throw ::gmx::test::TestException(e)

} // namespace test
} // namespace gmx

#endif
