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
 * \brief Stub for update and constraints class CPU implementation.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/mdlib/update_constrain_gpu.h"
#include "gromacs/utility/gmxassert.h"

#if !GMX_GPU_CUDA && !GMX_GPU_SYCL

namespace gmx
{

class UpdateConstrainGpu::Impl
{
};

UpdateConstrainGpu::UpdateConstrainGpu(const t_inputrec& /* ir   */,
                                       const gmx_mtop_t& /* mtop */,
                                       const int /* numTempScaleValues */,
                                       const DeviceContext& /* deviceContext */,
                                       const DeviceStream& /* deviceStream */,
                                       gmx_wallcycle* /*wcycle*/) :
    impl_(nullptr)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for UpdateConstrain was called instead of the correct implementation.");
}

UpdateConstrainGpu::~UpdateConstrainGpu() = default;

void UpdateConstrainGpu::integrate(GpuEventSynchronizer* /* fReadyOnDevice */,
                                   const real /* dt */,
                                   const bool /* updateVelocities */,
                                   const bool /* computeVirial */,
                                   tensor /* virialScaled */,
                                   const bool /* doTemperatureScaling */,
                                   gmx::ArrayRef<const t_grp_tcstat> /* tcstat */,
                                   const bool /* doParrinelloRahman */,
                                   const float /* dtPressureCouple */,
                                   const Matrix3x3& /* prVelocityScalingMatrix*/)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for UpdateConstrain was called instead of the correct implementation.");
}

void UpdateConstrainGpu::scaleCoordinates(const Matrix3x3& /* scalingMatrix */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for UpdateConstrain was called instead of the correct implementation.");
}

void UpdateConstrainGpu::scaleVelocities(const Matrix3x3& /* scalingMatrix */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for UpdateConstrain was called instead of the correct implementation.");
}

void UpdateConstrainGpu::set(DeviceBuffer<RVec> /* d_x */,
                             DeviceBuffer<RVec> /* d_v */,
                             const DeviceBuffer<RVec> /* d_f */,
                             const InteractionDefinitions& /* idef */,
                             const t_mdatoms& /* md */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for UpdateConstrain was called instead of the correct implementation.");
}

void UpdateConstrainGpu::setPbc(const PbcType /* pbcType */, const matrix /* box */)
{
    GMX_ASSERT(!impl_,
               "A CPU stub for UpdateConstrain was called instead of the correct implementation.");
}

GpuEventSynchronizer* UpdateConstrainGpu::xUpdatedOnDeviceEvent()
{
    GMX_ASSERT(!impl_,
               "A CPU stub for UpdateConstrain was called instead of the correct implementation.");
    return nullptr;
}

bool UpdateConstrainGpu::isNumCoupledConstraintsSupported(const gmx_mtop_t& /* mtop */)
{
    return false;
}


bool UpdateConstrainGpu::areConstraintsSupported()
{
    return false;
}

} // namespace gmx

#endif /* !GMX_GPU_CUDA && !GMX_GPU_SYCL */
