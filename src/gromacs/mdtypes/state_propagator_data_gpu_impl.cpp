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
 * \brief The CPU stub for the state propagator data class.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_mdtypes
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/mdtypes/state_propagator_data_gpu.h"

#if !GMX_GPU
namespace gmx
{

class StatePropagatorDataGpu::Impl
{
};

StatePropagatorDataGpu::StatePropagatorDataGpu(const DeviceStreamManager& /* deviceStreamManager */,
                                               GpuApiCallBehavior /* transferKind    */,
                                               int /* allocationBlockSizeDivisor */,
                                               gmx_wallcycle* /*   wcycle */) :
    impl_(nullptr)
{
}

StatePropagatorDataGpu::StatePropagatorDataGpu(const DeviceStream* /* pmeStream       */,
                                               const DeviceContext& /* deviceContext   */,
                                               GpuApiCallBehavior /* transferKind    */,
                                               int /* allocationBlockSizeDivisor */,
                                               gmx_wallcycle* /*   wcycle */) :
    impl_(nullptr)
{
}

StatePropagatorDataGpu::StatePropagatorDataGpu(StatePropagatorDataGpu&& /* other */) noexcept = default;

StatePropagatorDataGpu& StatePropagatorDataGpu::operator=(StatePropagatorDataGpu&& /* other */) noexcept = default;

StatePropagatorDataGpu::~StatePropagatorDataGpu() = default;

void StatePropagatorDataGpu::reinit(int /* numAtomsLocal */, int /* numAtomsAll   */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

std::tuple<int, int> StatePropagatorDataGpu::getAtomRangesFromAtomLocality(AtomLocality /* atomLocality */) const
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return std::make_tuple(0, 0);
}

DeviceBuffer<RVec> StatePropagatorDataGpu::getCoordinates()
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return {};
}

GpuEventSynchronizer* StatePropagatorDataGpu::getCoordinatesReadyOnDeviceEvent(
        AtomLocality /* atomLocality */,
        const SimulationWorkload& /* simulationWork */,
        const StepWorkload& /* stepWork       */,
        GpuEventSynchronizer* /* gpuCoordinateHaloLaunched */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return nullptr;
}

void StatePropagatorDataGpu::waitCoordinatesCopiedToDevice(AtomLocality /* atomLocality */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::consumeCoordinatesCopiedToDeviceEvent(AtomLocality /* atomLocality */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::resetCoordinatesCopiedToDeviceEvent(AtomLocality /* atomLocality */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::setXUpdatedOnDeviceEvent(GpuEventSynchronizer* /* xUpdatedOnDeviceEvent */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::setXUpdatedOnDeviceEventExpectedConsumptionCount(int /* expectedConsumptionCount */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::setFReadyOnDeviceEventExpectedConsumptionCount(AtomLocality /*atomLocality*/,
                                                                            int /*expectedConsumptionCount*/)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}


void StatePropagatorDataGpu::copyCoordinatesToGpu(const gmx::ArrayRef<const gmx::RVec> /* h_x */,
                                                  AtomLocality /* atomLocality */,
                                                  int /* expectedConsumptionCount */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::waitCoordinatesReadyOnHost(AtomLocality /* atomLocality */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::copyCoordinatesFromGpu(gmx::ArrayRef<gmx::RVec> /* h_x          */,
                                                    AtomLocality /* atomLocality */,
                                                    GpuEventSynchronizer* /*dependency */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

DeviceBuffer<RVec> StatePropagatorDataGpu::getVelocities()
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return {};
}

void StatePropagatorDataGpu::copyVelocitiesToGpu(const gmx::ArrayRef<const gmx::RVec> /* h_v */,
                                                 AtomLocality /* atomLocality */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::copyVelocitiesFromGpu(gmx::ArrayRef<gmx::RVec> /* h_v          */,
                                                   AtomLocality /* atomLocality */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::waitVelocitiesReadyOnHost(AtomLocality /* atomLocality */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}


DeviceBuffer<RVec> StatePropagatorDataGpu::getForces()
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return {};
}

void StatePropagatorDataGpu::copyForcesToGpu(const gmx::ArrayRef<const gmx::RVec> /* h_f          */,
                                             AtomLocality /* atomLocality */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::clearForcesOnGpu(AtomLocality /* atomLocality */,
                                              GpuEventSynchronizer* /* dependency */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

GpuEventSynchronizer* StatePropagatorDataGpu::getLocalForcesReadyOnDeviceEvent(StepWorkload /* stepWork */,
                                                                               SimulationWorkload /* simulationWork */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return nullptr;
}

GpuEventSynchronizer* StatePropagatorDataGpu::fReducedOnDevice(AtomLocality /*atomLocality*/)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return nullptr;
}

void StatePropagatorDataGpu::consumeForcesReducedOnDeviceEvent(AtomLocality /*atomLocality*/)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

GpuEventSynchronizer* StatePropagatorDataGpu::fReadyOnDevice(AtomLocality /*atomLocality*/)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return nullptr;
}

void StatePropagatorDataGpu::copyForcesFromGpu(gmx::ArrayRef<gmx::RVec> /* h_f          */,
                                               AtomLocality /* atomLocality */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

void StatePropagatorDataGpu::waitForcesReadyOnHost(AtomLocality /* atomLocality */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}


const DeviceStream* StatePropagatorDataGpu::getUpdateStream()
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return nullptr;
}

int StatePropagatorDataGpu::numAtomsLocal() const
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return 0;
}

int StatePropagatorDataGpu::numAtomsAll() const
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
    return 0;
}

void StatePropagatorDataGpu::waitCoordinatesUpdatedOnDevice()
{
    GMX_ASSERT(!impl_,
               "A CPU stub method from GPU state propagator data was called instead of one from "
               "GPU implementation.");
}

} // namespace gmx

#endif // !GMX_GPU
