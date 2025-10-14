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

#ifndef GMX_TESTUTILS_TEST_DEVICE_H
#define GMX_TESTUTILS_TEST_DEVICE_H

/*! \internal \file
 * \brief
 * Describes test environment class which performs GPU device enumeration for unit tests.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_testutils
 */

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gromacs/utility/gmxassert.h"

class DeviceContext;
struct DeviceInformation;
class DeviceStream;

namespace gmx
{
namespace test
{

/*! \internal \brief
 * A structure to describe a hardware context that persists over the lifetime
 * of the test binary.
 */
class TestDevice
{
public:
    //! Returns a human-readable context description line
    std::string description() const;
    //! Returns a numerical ID for the device
    int id() const;
    //! Returns the device info pointer
    const DeviceInformation& deviceInfo() const;
    //! Get the device context
    const DeviceContext& deviceContext() const;
    //! Get the device stream
    const DeviceStream& deviceStream() const;
    //! Set the device as currently active
    void activate() const;
    //! Creates the device context and stream for tests on the GPU
    TestDevice(const char* description, const DeviceInformation& deviceInfo);
    //! Destructor
    ~TestDevice();

private:
    //! Implementation type.
    class Impl;
    //! Implementation object.
    std::unique_ptr<Impl> impl_;
};

} // namespace test
} // namespace gmx

#endif // GMX_TESTUTILS_TEST_DEVICE_H
