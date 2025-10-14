/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * \brief Implements backend-specific functions of the update-constraints in CUDA.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "update_constrain_gpu_internal.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/gpu_utils/vectype_ops.cuh"

namespace gmx
{

/*!\brief Number of CUDA threads in a block
 *
 * \todo Check if using smaller block size will lead to better performance.
 */
constexpr static int c_threadsPerBlock = 256;
//! Maximum number of threads in a block (for __launch_bounds__)
constexpr static int c_maxThreadsPerBlock = c_threadsPerBlock;

__launch_bounds__(c_maxThreadsPerBlock) __global__
        static void scaleCoordinates_kernel(const int numAtoms,
                                            float3* __restrict__ gm_x,
                                            const ScalingMatrix scalingMatrix)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex < numAtoms)
    {
        float3 x = gm_x[threadIndex];

        x.x = scalingMatrix.xx * x.x + scalingMatrix.yx * x.y + scalingMatrix.zx * x.z;
        x.y = scalingMatrix.yy * x.y + scalingMatrix.zy * x.z;
        x.z = scalingMatrix.zz * x.z;

        gm_x[threadIndex] = x;
    }
}

void launchScaleCoordinatesKernel(const int            numAtoms,
                                  DeviceBuffer<Float3> d_coordinates,
                                  const ScalingMatrix& mu,
                                  const DeviceStream&  deviceStream)
{
    KernelLaunchConfig kernelLaunchConfig;

    kernelLaunchConfig.blockSize[0]     = c_threadsPerBlock;
    kernelLaunchConfig.blockSize[1]     = 1;
    kernelLaunchConfig.blockSize[2]     = 1;
    kernelLaunchConfig.sharedMemorySize = 0;

    kernelLaunchConfig.gridSize[0] = (numAtoms + c_threadsPerBlock - 1) / c_threadsPerBlock;

    const auto kernelArgs = prepareGpuKernelArguments(
            scaleCoordinates_kernel, kernelLaunchConfig, &numAtoms, asFloat3Pointer(&d_coordinates), &mu);
    launchGpuKernel(scaleCoordinates_kernel,
                    kernelLaunchConfig,
                    deviceStream,
                    nullptr,
                    "scaleCoordinates_kernel",
                    kernelArgs);
    // TODO: Although this only happens on the pressure coupling steps, this synchronization
    //       can affect the performance if nstpcouple is small. See Issue #4018
    deviceStream.synchronize();
}

} // namespace gmx
