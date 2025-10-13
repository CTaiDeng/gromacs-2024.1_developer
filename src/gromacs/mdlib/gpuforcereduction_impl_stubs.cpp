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
 *
 * \brief May be used to implement force reduction interfaces for non-GPU builds.
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_mdlib
 */

#include "gmxpre.h"

#include "config.h"

#include "gpuforcereduction.h"

#if !HAVE_GPU_FORCE_REDUCTION

namespace gmx
{

class GpuForceReduction::Impl
{
};

GpuForceReduction::GpuForceReduction(const DeviceContext& /* deviceContext */,
                                     const DeviceStream& /* deviceStream */,
                                     gmx_wallcycle* /*wcycle*/) :
    impl_(nullptr)
{
    GMX_RELEASE_ASSERT(false, "A CPU stub has been called instead of the correct implementation.");
}

// NOLINTNEXTLINE readability-convert-member-functions-to-static
void GpuForceReduction::reinit(DeviceBuffer<RVec> /*baseForcePtr*/,
                               const int /*numAtoms*/,
                               ArrayRef<const int> /*cell*/,
                               const int /*atomStart*/,
                               const bool /*accumulate*/,
                               GpuEventSynchronizer* /*completionMarker*/)
{
    GMX_RELEASE_ASSERT(false, "A CPU stub has been called instead of the correct implementation.");
}

// NOLINTNEXTLINE readability-convert-member-functions-to-static
void GpuForceReduction::registerNbnxmForce(DeviceBuffer<RVec> /* forcePtr */)
{
    GMX_RELEASE_ASSERT(false, "A CPU stub has been called instead of the correct implementation.");
}

// NOLINTNEXTLINE readability-convert-member-functions-to-static
void GpuForceReduction::registerRvecForce(DeviceBuffer<gmx::RVec> /* forcePtr */)
{
    GMX_RELEASE_ASSERT(false, "A CPU stub has been called instead of the correct implementation.");
}

// NOLINTNEXTLINE readability-convert-member-functions-to-static
void GpuForceReduction::registerForcesReadyNvshmemFlags(DeviceBuffer<uint64_t> /* forceSyncObjPtr */)
{
    GMX_RELEASE_ASSERT(false, "A CPU stub has been called instead of the correct implementation.");
}

// NOLINTNEXTLINE readability-convert-member-functions-to-static
void GpuForceReduction::addDependency(GpuEventSynchronizer* const /* dependency */)
{
    GMX_RELEASE_ASSERT(false, "A CPU stub has been called instead of the correct implementation.");
}

// NOLINTNEXTLINE readability-convert-member-functions-to-static
void GpuForceReduction::execute()
{
    GMX_RELEASE_ASSERT(false, "A CPU stub has been called instead of the correct implementation.");
}

GpuForceReduction::~GpuForceReduction() = default;

} // namespace gmx

#endif /* !HAVE_GPU_FORCE_REDUCTION */
