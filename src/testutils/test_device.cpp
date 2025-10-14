/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

#include "testutils/test_device.h"

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/hardware/device_information.h"

namespace gmx
{
namespace test
{

class TestDevice::Impl
{
public:
    Impl(const char* description, const DeviceInformation& deviceInfo);
    ~Impl();
    //! Returns a human-readable context description line
    std::string description() const { return description_; }
    //! Returns the device info pointer
    const DeviceInformation& deviceInfo() const { return deviceContext_.deviceInfo(); }
    //! Get the device context
    const DeviceContext& deviceContext() const { return deviceContext_; }
    //! Get the device stream
    const DeviceStream& deviceStream() const { return deviceStream_; }

private:
    //! Readable description
    std::string description_;
    //! Device context
    DeviceContext deviceContext_;
    //! Device stream
    DeviceStream deviceStream_;
};

TestDevice::Impl::Impl(const char* description, const DeviceInformation& deviceInfo) :
    description_(description),
    deviceContext_(deviceInfo),
    deviceStream_(deviceContext_, DeviceStreamPriority::Normal, false)
{
}

TestDevice::Impl::~Impl() = default;

TestDevice::TestDevice(const char* description, const DeviceInformation& deviceInfo) :
    impl_(new Impl(description, deviceInfo))
{
}

TestDevice::~TestDevice() = default;

std::string TestDevice::description() const
{
    return impl_->description();
}

int TestDevice::id() const
{
    return deviceInfo().id;
}

const DeviceInformation& TestDevice::deviceInfo() const
{
    return impl_->deviceInfo();
}

const DeviceContext& TestDevice::deviceContext() const
{
    return impl_->deviceContext();
}

const DeviceStream& TestDevice::deviceStream() const
{
    return impl_->deviceStream();
}

void TestDevice::activate() const
{
    deviceContext().activate();
}

} // namespace test
} // namespace gmx
