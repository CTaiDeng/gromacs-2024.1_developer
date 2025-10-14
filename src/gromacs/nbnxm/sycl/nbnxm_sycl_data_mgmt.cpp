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
 *  \brief
 *  Stubs of functions that must be defined by nbnxm sycl implementation.
 *
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "gromacs/gpu_utils/pmalloc.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/nbnxm_gpu_data_mgmt.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/exceptions.h"

#include "nbnxm_sycl_types.h"

namespace Nbnxm
{

void gpu_init_platform_specific(NbnxmGpu* /* nb */)
{
    // Nothing specific in SYCL
}

void gpu_free_platform_specific(NbnxmGpu* /* nb */)
{
    // Nothing specific in SYCL
}

int gpu_min_ci_balanced(NbnxmGpu* nb)
{
    // SYCL-TODO: Logic and magic values taken from OpenCL
    static constexpr unsigned int balancedFactor = 50;
    if (nb == nullptr)
    {
        return 0;
    }
    const DeviceInformation& deviceInfo = nb->deviceContext_->deviceInfo();
    const sycl::device       device     = deviceInfo.syclDevice;
    const int numComputeUnits           = device.get_info<sycl::info::device::max_compute_units>();
    const int numComputeUnitsFactor     = getDeviceComputeUnitFactor(deviceInfo);
    return balancedFactor * numComputeUnits / numComputeUnitsFactor;
}

} // namespace Nbnxm
