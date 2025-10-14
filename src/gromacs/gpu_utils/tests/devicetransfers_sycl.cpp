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
 * \brief Defines helper functionality for device transfers for tests
 * for GPU host allocator.
 *
 * \author Andrey Alekseenko <al42and@gmail.com>
 */
#include "gmxpre.h"

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/gmxsycl.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#include "devicetransfers.h"

namespace gmx
{

void doDeviceTransfers(const DeviceContext&     deviceContext,
                       const DeviceInformation& deviceInfo,
                       ArrayRef<const char>     input,
                       ArrayRef<char>           output)
{
    GMX_RELEASE_ASSERT(input.size() == output.size(), "Input and output must have matching size");

    try
    {
        sycl::queue syclQueue(deviceContext.context(), deviceInfo.syclDevice);

        sycl::global_ptr<char> d_buf = sycl::malloc_device<char>(input.size(), syclQueue);

        syclQueue.memcpy(d_buf, input.data(), input.size()).wait_and_throw();

        syclQueue.memcpy(output.data(), d_buf, input.size()).wait_and_throw();

        sycl::free(d_buf, syclQueue);
    }
    catch (sycl::exception& e)
    {
        GMX_THROW(InternalError(
                formatString("Failure while checking data transfer, error was %s", e.what())));
    }
}

} // namespace gmx
