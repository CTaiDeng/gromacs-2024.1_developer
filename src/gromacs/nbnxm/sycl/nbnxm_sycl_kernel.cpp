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
 *  \brief
 *  NBNXM SYCL kernels
 *
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "nbnxm_sycl_kernel.h"

#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/utility/template_mp.h"

#include "nbnxm_sycl_types.h"

namespace Nbnxm
{

static int getNbnxmSubGroupSize(const DeviceInformation& deviceInfo)
{
    if (deviceInfo.supportedSubGroupSizesSize == 1)
    {
        return deviceInfo.supportedSubGroupSizesData[0];
    }
    else if (deviceInfo.supportedSubGroupSizesSize > 1)
    {
        switch (deviceInfo.deviceVendor)
        {
            /* For Intel, choose 8 for 4x4 clusters, and 32 for 8x8 clusters.
             * The optimal one depends on the hardware, but we cannot choose c_nbnxnGpuClusterSize
             * at runtime anyway yet. */
            case DeviceVendor::Intel:
                return c_nbnxnGpuClusterSize * c_nbnxnGpuClusterSize / c_nbnxnGpuClusterpairSplit;
            default:
                GMX_RELEASE_ASSERT(false, "Flexible sub-groups only supported for Intel GPUs");
                return 0;
        }
    }
    else
    {
        GMX_RELEASE_ASSERT(false, "Device has no known supported sub-group sizes");
        return 0;
    }
}

template<int subGroupSize, bool doPruneNBL, bool doCalcEnergies>
void launchNbnxmKernelHelper(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const InteractionLocality iloc);

// clang-format off
#if SYCL_NBNXM_SUPPORTS_SUBGROUP_SIZE_8
extern template void launchNbnxmKernelHelper<8, false, false>(NbnxmGpu* nb,const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
extern template void launchNbnxmKernelHelper<8, false, true>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
extern template void launchNbnxmKernelHelper<8, true, false>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
extern template void launchNbnxmKernelHelper<8, true, true>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
#endif
#if SYCL_NBNXM_SUPPORTS_SUBGROUP_SIZE_32
extern template void launchNbnxmKernelHelper<32, false, false>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
extern template void launchNbnxmKernelHelper<32, false, true>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
extern template void launchNbnxmKernelHelper<32, true, false>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
extern template void launchNbnxmKernelHelper<32, true, true>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
#endif
#if SYCL_NBNXM_SUPPORTS_SUBGROUP_SIZE_64
extern template void launchNbnxmKernelHelper<64, false, false>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
extern template void launchNbnxmKernelHelper<64, false, true>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
extern template void launchNbnxmKernelHelper<64, true, false>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
extern template void launchNbnxmKernelHelper<64, true, true>(NbnxmGpu* nb, const gmx::StepWorkload&  stepWork, const InteractionLocality iloc);
#endif
// clang-format on

template<int subGroupSize>
void launchNbnxmKernel(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const InteractionLocality iloc)
{
    const bool doPruneNBL     = (nb->plist[iloc]->haveFreshList && !nb->didPrune[iloc]);
    const bool doCalcEnergies = stepWork.computeEnergy;

    gmx::dispatchTemplatedFunction(
            [&](auto doPruneNBL_, auto doCalcEnergies_) {
                launchNbnxmKernelHelper<subGroupSize, doPruneNBL_, doCalcEnergies_>(nb, stepWork, iloc);
            },
            doPruneNBL,
            doCalcEnergies);
}

void launchNbnxmKernel(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const InteractionLocality iloc)
{
    const int subGroupSize = getNbnxmSubGroupSize(nb->deviceContext_->deviceInfo());
    switch (subGroupSize)
    {
        // Ensure any changes are in sync with device_management_sycl.cpp, nbnxm_sycl_kernel_body.h, and the #if above
#if SYCL_NBNXM_SUPPORTS_SUBGROUP_SIZE_8
        case 8: launchNbnxmKernel<8>(nb, stepWork, iloc); break;
#endif
#if SYCL_NBNXM_SUPPORTS_SUBGROUP_SIZE_32
        case 32: launchNbnxmKernel<32>(nb, stepWork, iloc); break;
#endif
#if SYCL_NBNXM_SUPPORTS_SUBGROUP_SIZE_64
        case 64: launchNbnxmKernel<64>(nb, stepWork, iloc); break;
#endif
        default: GMX_RELEASE_ASSERT(false, "Unsupported sub-group size");
    }
}

} // namespace Nbnxm
