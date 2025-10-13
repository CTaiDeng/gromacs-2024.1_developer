/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * Declares gmx::test::LoggerTestHelper.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_testutils
 */
#ifndef GMX_TESTUTILS_LOGGERTEST_H
#define GMX_TESTUTILS_LOGGERTEST_H

#include <memory>

#include "gromacs/utility/logger.h"

namespace gmx
{

namespace test
{

/*! \libinternal \brief
 * Helper class for tests to check output written to a logger.
 *
 * \inlibraryapi
 * \ingroup module_testutils
 */
class LoggerTestHelper
{
public:
    LoggerTestHelper();
    ~LoggerTestHelper();

    //! Returns the logger to pass to code under test.
    const MDLogger& logger();

    /*! \brief
     * Expects a log entry at a given level matching a given regex.
     *
     * Currently, the order of the entries is not checked, and if this
     * method is called once for a log level, then it needs to be called
     * for all entries produced by the test.
     *
     * If not called for a log level, all entries for that level are
     * accepted.
     *
     * Note that this expectation should be set up before the logger is
     * used in the test.
     */
    void expectEntryMatchingRegex(gmx::MDLogger::LogLevel level, const char* re);

    /*! \brief
     * Expects that no log entries were made at a given level.
     *
     * If not called for a log level, all entries for that level are
     * accepted.
     *
     * Note that this expectation should be set up before the logger is
     * used in the test.
     */
    void expectNoEntries(gmx::MDLogger::LogLevel level);

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace test
} // namespace gmx

#endif
