/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 *
 * \brief May be used to implement PME-PP GPU comm interfaces for non-GPU builds.
 *
 * Currently, reports and exits if any of the interfaces are called.
 * Needed to satisfy compiler on systems, where CUDA is not available.
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/ewald/pme_force_sender_gpu.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/gmxassert.h"

#if !GMX_GPU_CUDA && !GMX_GPU_SYCL

namespace gmx
{

/*! \internal \brief Impl class stub. */
class PmeForceSenderGpu::Impl
{
};

/*!\brief Constructor stub. */
PmeForceSenderGpu::PmeForceSenderGpu(GpuEventSynchronizer* /*pmeForcesReady */,
                                     MPI_Comm /* comm     */,
                                     const DeviceContext& /* deviceContext */,
                                     gmx::ArrayRef<PpRanks> /* ppRanks */) :
    impl_(nullptr)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

PmeForceSenderGpu::~PmeForceSenderGpu() = default;

/*!\brief init PME-PP GPU communication stub */
void PmeForceSenderGpu::setForceSendBuffer(DeviceBuffer<RVec> /* d_f */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication initialization was called instead of the "
               "correct implementation.");
}

void PmeForceSenderGpu::sendFToPpPeerToPeer(int /* ppRank */,
                                            int /* numAtoms */,
                                            bool /* sendForcesDirectToPpGpu */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

void PmeForceSenderGpu::sendFToPpGpuAwareMpi(DeviceBuffer<RVec> /* sendbuf */,
                                             int /* offset */,
                                             int /* numBytes */,
                                             int /* ppRank */,
                                             MPI_Request* /* request */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

void PmeForceSenderGpu::waitForEvents()
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}


} // namespace gmx

#endif // !GMX_GPU_CUDA
