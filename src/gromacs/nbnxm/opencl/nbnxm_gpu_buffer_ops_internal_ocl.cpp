/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 *  \brief Define OpenCL implementation for transforming position coordinates from rvec to nbnxm layout.
 *
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/nbnxm_gpu_buffer_ops_internal.h"

namespace Nbnxm
{

void launchNbnxmKernelTransformXToXq(const Grid& /* grid */,
                                     NbnxmGpu* /* nb */,
                                     DeviceBuffer<Float3> /* d_x */,
                                     const DeviceStream& /* deviceStream */,
                                     unsigned int /* numColumnsMax */,
                                     int /* gridId */)
{
    GMX_RELEASE_ASSERT(false, "NBNXM buffer ops are not supported with OpenCL");
}

} // namespace Nbnxm
