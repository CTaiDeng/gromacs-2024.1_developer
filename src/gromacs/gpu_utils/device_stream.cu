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
 *
 * \brief Implements the DeviceStream for CUDA.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_gpu_utils
 */
#include "gmxpre.h"

#include "device_stream.h"

#include <cstdio>

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

DeviceStream::DeviceStream(const DeviceContext& /* deviceContext */,
                           DeviceStreamPriority priority,
                           const bool /* useTiming */)
{
    cudaError_t stat;

    if (priority == DeviceStreamPriority::Normal)
    {
        stat = cudaStreamCreate(&stream_);
        gmx::checkDeviceError(stat, "Could not create CUDA stream.");
    }
    else if (priority == DeviceStreamPriority::High)
    {
        // Note that the device we're running on does not have to
        // support priorities, because we are querying the priority
        // range, which in that case will be a single value.
        int highestPriority;
        stat = cudaDeviceGetStreamPriorityRange(nullptr, &highestPriority);
        gmx::checkDeviceError(stat, "Could not query CUDA stream priority range.");

        stat = cudaStreamCreateWithPriority(&stream_, cudaStreamDefault, highestPriority);
        gmx::checkDeviceError(stat, "Could not create CUDA stream with high priority.");
    }
}

DeviceStream::~DeviceStream()
{
    if (isValid())
    {
        cudaError_t stat = cudaStreamDestroy(stream_);
        if (stat != cudaSuccess)
        {
            // Don't throw in the destructor, just print a warning
            std::fprintf(stderr,
                         "Failed to release CUDA stream. %s\n",
                         gmx::getDeviceErrorString(stat).c_str());
        }
        stream_ = nullptr;
    }
}

cudaStream_t DeviceStream::stream() const
{
    return stream_;
}

bool DeviceStream::isValid() const
{
    return (stream_ != nullptr);
}

void DeviceStream::synchronize() const
{
    cudaError_t stat = cudaStreamSynchronize(stream_);
    GMX_RELEASE_ASSERT(stat == cudaSuccess,
                       ("cudaStreamSynchronize failed. " + gmx::getDeviceErrorString(stat)).c_str());
}

void issueClFlushInStream(const DeviceStream& /*deviceStream*/) {}
