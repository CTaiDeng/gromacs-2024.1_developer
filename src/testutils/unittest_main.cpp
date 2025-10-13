/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * main() for unit tests that use \ref module_testutils.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include <gtest/gtest.h>

#include "testutils/testinit.h"

#ifndef TEST_DATA_PATH
//! Path to test input data directory (needs to be set by the build system).
#    define TEST_DATA_PATH ""
#endif

#ifndef TEST_TEMP_PATH
//! Path to test output temporary directory (needs to be set by the build system).
#    define TEST_TEMP_PATH ""
#endif

#ifndef TEST_USES_MPI
//! Whether the test expects/supports running with multiple MPI ranks.
#    define TEST_USES_MPI false
#endif

#ifndef TEST_USES_HARDWARE_DETECTION
//! Whether the test expects/supports running with knowledge of the hardware.
#    define TEST_USES_HARDWARE_DETECTION false
#endif

#ifndef TEST_USES_DYNAMIC_REGISTRATION
//! Whether tests will be dynamically registered
#    define TEST_USES_DYNAMIC_REGISTRATION false
namespace gmx
{
namespace test
{
// Stub implementation for test suites that do not use dynamic
// registration.
void registerTestsDynamically() {}
} // namespace test
} // namespace gmx
#endif

/*! \brief
 * Initializes unit testing for \ref module_testutils.
 */
int main(int argc, char* argv[])
{
    // Calls ::testing::InitGoogleMock()
    ::gmx::test::initTestUtils(TEST_DATA_PATH,
                               TEST_TEMP_PATH,
                               TEST_USES_MPI,
                               TEST_USES_HARDWARE_DETECTION,
                               TEST_USES_DYNAMIC_REGISTRATION,
                               &argc,
                               &argv);
    int errcode = RUN_ALL_TESTS();
    ::gmx::test::finalizeTestUtils(TEST_USES_HARDWARE_DETECTION, TEST_USES_DYNAMIC_REGISTRATION);
    return errcode;
}
