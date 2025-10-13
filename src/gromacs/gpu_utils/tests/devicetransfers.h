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
 * \brief Declares helper functionality for device transfers for tests
 * for GPU host allocator.
 *
 * Undefined symbols in Google Test, GROMACS use of -Wundef, and the
 * implementation of FindCUDA.cmake and/or nvcc mean that no
 * compilation unit should include a gtest header while being compiled
 * by nvcc. None of -isystem, -Wno-undef, nor the pragma GCC
 * diagnostic work.
 *
 * Thus, this header isolates CUDA-specific functionality to its own
 * translation unit. The OpenCL and no-GPU implementations do not
 * require this separation, but do so for consistency.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 */
#ifndef GMX_GPU_UTILS_TESTS_DEVICETRANSFERS_H
#define GMX_GPU_UTILS_TESTS_DEVICETRANSFERS_H

struct DeviceInformation;
class DeviceContext;

namespace gmx
{
template<typename>
class ArrayRef;

/*! \brief Helper function for GPU test code to be platform agnostic.
 *
 * Transfers \c input to device 0, if present, and transfers it back
 * into \c output. Both sizes must match. If no devices are present,
 * do a simple host-side buffer copy instead.
 *
 * \throws InternalError  Upon any GPU API error condition. */
void doDeviceTransfers(const DeviceContext&     deviceContext,
                       const DeviceInformation& deviceInfo,
                       ArrayRef<const char>     input,
                       ArrayRef<char>           output);

} // namespace gmx

#endif
