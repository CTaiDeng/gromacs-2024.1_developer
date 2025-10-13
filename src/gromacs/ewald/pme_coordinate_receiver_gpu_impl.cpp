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
 * Needed to satisfy compiler when compiling without GPU support.
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/gmxassert.h"

#if !GMX_GPU_CUDA && !GMX_GPU_SYCL

namespace gmx
{

/*!\brief \internal Impl class stub. */
class PmeCoordinateReceiverGpu::Impl
{
};

/*!\brief Constructor stub. */
PmeCoordinateReceiverGpu::PmeCoordinateReceiverGpu(MPI_Comm /* comm */,
                                                   const DeviceContext& /* deviceContext */,
                                                   gmx::ArrayRef<PpRanks> /* ppRanks */) :
    impl_(nullptr)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

PmeCoordinateReceiverGpu::~PmeCoordinateReceiverGpu() = default;

/*!\brief init PME-PP GPU communication stub */
void PmeCoordinateReceiverGpu::reinitCoordinateReceiver(DeviceBuffer<RVec> /* d_x */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication initialization was called instead of the "
               "correct implementation.");
}

void PmeCoordinateReceiverGpu::receiveCoordinatesSynchronizerFromPpPeerToPeer(int /* ppRank */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

void PmeCoordinateReceiverGpu::launchReceiveCoordinatesFromPpGpuAwareMpi(DeviceBuffer<RVec> /* recvbuf */,
                                                                         int /* numAtoms */,
                                                                         int /* numBytes */,
                                                                         int /* ppRank */,
                                                                         int /* senderIndex */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

std::tuple<int, GpuEventSynchronizer*> PmeCoordinateReceiverGpu::receivePpCoordinateSendEvent(int /* pipelineStage */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
    return std::make_tuple(0, nullptr);
}

int PmeCoordinateReceiverGpu::waitForCoordinatesFromAnyPpRank()
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
    return 0;
}

DeviceStream* PmeCoordinateReceiverGpu::ppCommStream(int /* senderIndex */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
    return nullptr;
}

std::tuple<int, int> PmeCoordinateReceiverGpu::ppCommAtomRange(int /* senderIndex */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
    return std::make_tuple(0, 0);
}

int PmeCoordinateReceiverGpu::ppCommNumSenderRanks()
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
    return 0;
}

void PmeCoordinateReceiverGpu::insertAsDependencyIntoStream(int /*senderIndex*/, const DeviceStream& /*stream*/)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for PME-PP GPU communication was called instead of the correct "
               "implementation.");
}

} // namespace gmx

#endif // !GMX_GPU_CUDA
