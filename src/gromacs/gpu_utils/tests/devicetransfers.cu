/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * Undefined symbols in Google Test, GROMACS use of -Wundef, and the
 * implementation of FindCUDA.cmake and/or nvcc mean that no
 * compilation unit should include a gtest header while being compiled
 * by nvcc. None of -isystem, -Wno-undef, nor the pragma GCC
 * diagnostic work.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 */
#include "gmxpre.h"

#include "devicetransfers.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

void doDeviceTransfers(const DeviceContext& /*context*/,
                       const DeviceInformation& deviceInfo,
                       ArrayRef<const char>     input,
                       ArrayRef<char>           output)
{
    GMX_RELEASE_ASSERT(input.size() == output.size(), "Input and output must have matching size");
    cudaError_t status;

    int oldDeviceId;

    status = cudaGetDevice(&oldDeviceId);
    checkDeviceError(status, "Error while getting old device id.");
    status = cudaSetDevice(deviceInfo.id);
    checkDeviceError(status, "Error while setting device id to the first compatible GPU.");

    void* devicePointer;
    status = cudaMalloc(&devicePointer, input.size());
    checkDeviceError(status, "Error while creating buffer.");

    status = cudaMemcpy(devicePointer, input.data(), input.size(), cudaMemcpyHostToDevice);
    checkDeviceError(status, "Error while transferring host to device.");
    status = cudaMemcpy(output.data(), devicePointer, output.size(), cudaMemcpyDeviceToHost);
    checkDeviceError(status, "Error while transferring device to host.");

    status = cudaFree(devicePointer);
    checkDeviceError(status, "Error while releasing buffer.");

    status = cudaSetDevice(oldDeviceId);
    checkDeviceError(status, "Error while setting old device id.");
}

} // namespace gmx
