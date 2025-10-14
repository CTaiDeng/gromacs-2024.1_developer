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

#include "gromacs/ewald/pme_pp_comm_gpu.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxmpi.h"

#if !GMX_GPU_CUDA && !GMX_GPU_SYCL

namespace gmx
{

/*!\brief \internal Impl class stub. */
class PmePpCommGpu::Impl
{
};

/*!\brief Constructor stub. */
PmePpCommGpu::PmePpCommGpu(MPI_Comm /* comm */,
                           int /* pmeRank */,
                           gmx::HostVector<gmx::RVec>* /* pmeCpuForceBuffer */,
                           const DeviceContext& /* deviceContext */,
                           const DeviceStream& /* deviceStream */,
                           const bool /*useNvshmem*/) :
    impl_(nullptr)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

PmePpCommGpu::~PmePpCommGpu() = default;

/*!\brief init PME-PP GPU communication stub */
//NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void PmePpCommGpu::reinit(int /* size */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication initialization was called instead of the "
               "correct implementation.");
}

//NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void PmePpCommGpu::receiveForceFromPme(RVec* /* recvPtr */, int /* recvSize */, bool /* receivePmeForceToGpu */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

//NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void PmePpCommGpu::sendCoordinatesToPmeFromGpu(DeviceBuffer<RVec> /* sendPtr */,
                                               int /* sendSize */,
                                               GpuEventSynchronizer* /* coordinatesOnDeviceEvent */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

//NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void PmePpCommGpu::sendCoordinatesToPmeFromCpu(RVec* /* sendPtr */, int /* sendSize */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

//NOLINTNEXTLINE(readability-convert-member-functions-to-static)
DeviceBuffer<gmx::RVec> PmePpCommGpu::getGpuForceStagingPtr()
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
    return DeviceBuffer<gmx::RVec>{};
}

//NOLINTNEXTLINE(readability-convert-member-functions-to-static)
GpuEventSynchronizer* PmePpCommGpu::getForcesReadySynchronizer()
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
    return nullptr;
}

DeviceBuffer<uint64_t> PmePpCommGpu::getGpuForcesSyncObj()
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
    return nullptr;
}


} // namespace gmx

#endif // !GMX_GPU_CUDA && !GMX_GPU_SYCL
