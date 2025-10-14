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
 * \brief May be used to implement Domdec CUDA interfaces for non-GPU builds.
 *
 * Currently, reports and exits if any of the interfaces are called.
 * Needed to satisfy compiler on systems, where CUDA is not available.
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_domdec
 */
#include "gmxpre.h"

#include "config.h"

#include <utility>

#include "gromacs/domdec/gpuhaloexchange.h"
#include "gromacs/utility/gmxassert.h"

#if !GMX_GPU_CUDA && !GMX_GPU_SYCL

namespace gmx
{

/*!\brief Impl class stub. */
class GpuHaloExchange::Impl
{
};

/*!\brief Constructor stub. */
GpuHaloExchange::GpuHaloExchange(gmx_domdec_t* /* dd */,
                                 int /* dimIndex */,
                                 MPI_Comm /* mpi_comm_mysim */,
                                 const DeviceContext& /* deviceContext */,
                                 int /*pulse */,
                                 gmx_wallcycle* /*wcycle*/) :
    impl_(nullptr)
{
    GMX_ASSERT(false,
               "A CPU stub for GPU Halo Exchange was called insted of the correct implementation.");
}

GpuHaloExchange::~GpuHaloExchange() = default;

GpuHaloExchange::GpuHaloExchange(GpuHaloExchange&&) noexcept = default;

GpuHaloExchange& GpuHaloExchange::operator=(GpuHaloExchange&& other) noexcept
{
    std::swap(impl_, other.impl_);
    return *this;
}

/*!\brief init halo exhange stub. */
void GpuHaloExchange::reinitHalo(DeviceBuffer<RVec> /* d_coordinatesBuffer */,
                                 DeviceBuffer<RVec> /* d_forcesBuffer */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for GPU Halo Exchange was called insted of the correct implementation.");
}

/*!\brief apply X halo exchange stub. */
GpuEventSynchronizer* GpuHaloExchange::communicateHaloCoordinates(const matrix /* box */,
                                                                  GpuEventSynchronizer* /*dependencyEvent*/)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for GPU Halo Exchange exchange was called insted of the correct "
               "implementation.");
    return nullptr;
}

/*!\brief apply F halo exchange stub. */
void GpuHaloExchange::communicateHaloForces(bool /* accumulateForces */,
                                            FixedCapacityVector<GpuEventSynchronizer*, 2>* /*dependencyEvents*/)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for GPU Halo Exchange was called insted of the correct implementation.");
}

/*!\brief get forces ready on device event stub. */
GpuEventSynchronizer* GpuHaloExchange::getForcesReadyOnDeviceEvent()
{
    GMX_ASSERT(!impl_,
               "A CPU stub for GPU Halo Exchange was called insted of the correct implementation.");
    return nullptr;
}

} // namespace gmx

#endif // !GMX_GPU_CUDA && !GMX_GPU_SYCL
