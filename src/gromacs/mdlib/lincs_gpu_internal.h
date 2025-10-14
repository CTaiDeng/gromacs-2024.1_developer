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
 * \brief Declare backend-specific LINCS GPU functions
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_mdlib
 */
#ifndef GMX_MDLIB_LINCS_GPU_INTERNAL_H
#define GMX_MDLIB_LINCS_GPU_INTERNAL_H

#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gputraits.h"

class DeviceStream;

namespace gmx
{

struct LincsGpuKernelParameters;

//! Number of threads in a GPU block
constexpr static int c_threadsPerBlock = 256;

/*! \brief Backend-specific function to launch LINCS kernel.
 *
 * \param kernelParams LINCS parameters.
 * \param d_x Initial coordinates before the integration.
 * \param d_xp Coordinates after the integration which will be updated.
 * \param updateVelocities Whether to also update velocities.
 * \param d_v Velocities to update (ignored if \p updateVelocities is \c false).
 * \param invdt Reciprocal of timestep.
 * \param computeVirial Whether to compute the virial.
 * \param deviceStream Device stream for kernel launch.
 */
void launchLincsGpuKernel(LincsGpuKernelParameters*   kernelParams,
                          const DeviceBuffer<Float3>& d_x,
                          DeviceBuffer<Float3>        d_xp,
                          bool                        updateVelocities,
                          DeviceBuffer<Float3>        d_v,
                          real                        invdt,
                          bool                        computeVirial,
                          const DeviceStream&         deviceStream);

} // namespace gmx

#endif // GMX_MDLIB_LINCS_GPU_INTERNAL_H
