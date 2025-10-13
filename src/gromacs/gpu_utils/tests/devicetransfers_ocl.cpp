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
 * \brief Defines helper functionality for device transfers for tests
 * for GPU host allocator.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 */
#include "gmxpre.h"

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/gmxopencl.h"
#include "gromacs/gpu_utils/oclutils.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#include "devicetransfers.h"

namespace gmx
{
namespace
{

/*! \brief Help give useful diagnostics about error \c status while doing \c message.
 *
 * \throws InternalError  If status indicates failure, supplying
 *                        descriptive text from \c message. */
void throwUponFailure(cl_int status, const char* message)
{
    if (status != CL_SUCCESS)
    {
        GMX_THROW(InternalError(formatString(
                "Failure while %s, error was %s", message, ocl_get_error_string(status).c_str())));
    }
}

} // namespace

void doDeviceTransfers(const DeviceContext&     deviceContext,
                       const DeviceInformation& deviceInfo,
                       ArrayRef<const char>     input,
                       ArrayRef<char>           output)
{
    GMX_RELEASE_ASSERT(input.size() == output.size(), "Input and output must have matching size");

    cl_int status;

    auto deviceId     = deviceInfo.oclDeviceId;
    auto context      = deviceContext.context();
    auto commandQueue = clCreateCommandQueue(context, deviceId, 0, &status);
    throwUponFailure(status, "creating command queue");

    auto devicePointer = clCreateBuffer(context, CL_MEM_READ_WRITE, input.size(), nullptr, &status);
    throwUponFailure(status, "creating buffer");

    status = clEnqueueWriteBuffer(
            commandQueue, devicePointer, CL_TRUE, 0, input.size(), input.data(), 0, nullptr, nullptr);
    throwUponFailure(status, "transferring host to device");
    status = clEnqueueReadBuffer(
            commandQueue, devicePointer, CL_TRUE, 0, output.size(), output.data(), 0, nullptr, nullptr);
    throwUponFailure(status, "transferring device to host");

    status = clReleaseMemObject(devicePointer);
    throwUponFailure(status, "releasing buffer");
    status = clReleaseCommandQueue(commandQueue);
    throwUponFailure(status, "releasing command queue");
}

} // namespace gmx
