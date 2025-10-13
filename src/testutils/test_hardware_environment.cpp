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

/*! \internal \file
 * \brief
 * Implements test environment class which performs hardware enumeration for unit tests.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_testutils
 */

#include "gmxpre.h"

#include "testutils/test_hardware_environment.h"

#include <memory>
#include <mutex>

#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/hardware/detecthardware.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/physicalnodecommunicator.h"

namespace gmx
{
namespace test
{

//! Mutex for making the test hardware environment
static std::mutex s_testHardwareEnvironmentMutex;
//! The test hardware environment
static std::unique_ptr<TestHardwareEnvironment> s_testHardwareEnvironment;

const TestHardwareEnvironment* getTestHardwareEnvironment()
{
    if (!s_testHardwareEnvironment)
    {
        // Construct and fill the environment
        std::lock_guard<std::mutex> lock(s_testHardwareEnvironmentMutex);
        s_testHardwareEnvironment = std::make_unique<TestHardwareEnvironment>();
    }
    return s_testHardwareEnvironment.get();
}

TestHardwareEnvironment::TestHardwareEnvironment() :
    hardwareInfo_(gmx_detect_hardware(PhysicalNodeCommunicator{ MPI_COMM_WORLD, gmx_physicalnode_id_hash() },
                                      MPI_COMM_WORLD))
{
    // Following the ::testing::Environment protocol
    this->SetUp();

    // Constructing contexts for all compatible GPUs - will be empty on non-GPU builds
    for (const DeviceInformation& compatibleDeviceInfo : getCompatibleDevices(hardwareInfo_->deviceInfoList))
    {
        setActiveDevice(compatibleDeviceInfo);
        std::string description = getDeviceInformationString(compatibleDeviceInfo);
        testDeviceList_.emplace_back(std::make_unique<TestDevice>(description.c_str(), compatibleDeviceInfo));
    }
}

// static
void TestHardwareEnvironment::gmxSetUp()
{
    // Ensure the environment has been constructed
    getTestHardwareEnvironment();
}

// static
void TestHardwareEnvironment::gmxTearDown()
{
    std::lock_guard<std::mutex> lock(s_testHardwareEnvironmentMutex);
    if (!s_testHardwareEnvironment)
    {
        return;
    }
    s_testHardwareEnvironment->testDeviceList_.clear();
    s_testHardwareEnvironment->hardwareInfo_.reset();
}

} // namespace test
} // namespace gmx
