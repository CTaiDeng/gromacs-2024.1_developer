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

#ifndef GMX_TESTUTILS_TEST_HARDWARE_ENVIRONMENT_H
#define GMX_TESTUTILS_TEST_HARDWARE_ENVIRONMENT_H

/*! \internal \file
 * \brief
 * Describes test environment class which performs hardware enumeration for unit tests.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_testutils
 */

#include <map>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/utility/gmxassert.h"

#include "testutils/test_device.h"

struct gmx_hw_info_t;

namespace gmx
{
namespace test
{

/*! \internal
 *  \brief This class performs one-time test initialization, enumerating the hardware
 *
 * Note that this class follows the approach of GoogleTest's
 * Environment managers but we don't actually add it with
 * AddGlobalTestEnvironment. That's because it would not
 * work for the cases when we need information about the hardware
 * present at run time to register tests dynamically. So instead we
 * arrange for ::gmx::test::initTestUtils() and
 * ::gmx::test::finalizeTestUtils() to call
 * setupTestHardwareEnvironment() and
 * tearDownTestHardwareEnvironment() manually.
 */
class TestHardwareEnvironment : public ::testing::Environment
{
private:
    //! General hardware info
    std::unique_ptr<gmx_hw_info_t> hardwareInfo_;
    //! Storage of hardware contexts
    std::vector<std::unique_ptr<TestDevice>> testDeviceList_;

public:
    TestHardwareEnvironment();
    //! Get available hardware contexts.
    const std::vector<std::unique_ptr<TestDevice>>& getTestDeviceList() const
    {
        return testDeviceList_;
    }
    //! Whether the available hardware has any compatible devices
    bool hasCompatibleDevices() const { return !testDeviceList_.empty(); }
    //! Get available hardware information.
    const gmx_hw_info_t* hwinfo() const { return hardwareInfo_.get(); }

    /*! \brief Set up the test hardware environment
     *
     * We'd like to use GoogleTest's environment setup for this, but when
     * registering test dynamically we need the information before
     * GoogleTest would make it available. So instead we always handle it
     * ourselves, for simplicity.
     *
     * Should only be called once per test binary. */
    static void gmxSetUp();
    //! Tear down the test hardware environment
    static void gmxTearDown();
};

//! Get the global test environment
const TestHardwareEnvironment* getTestHardwareEnvironment();

} // namespace test
} // namespace gmx
#endif // GMX_TESTUTILS_TEST_HARDWARE_ENVIRONMENT_H
