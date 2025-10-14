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
 * \brief Implements SETTLE using GPU
 *
 * This file contains implementation for the data management of GPU version of SETTLE constraints algorithm.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "settle_gpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>

#include <algorithm>

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/constraint_gpu_helpers.h"
#include "gromacs/mdlib/settle_gpu_internal.h"
#include "gromacs/pbcutil/pbc.h"

namespace gmx
{

void SettleGpu::apply(const DeviceBuffer<Float3>& d_x,
                      DeviceBuffer<Float3>        d_xp,
                      const bool                  updateVelocities,
                      DeviceBuffer<Float3>        d_v,
                      const real                  invdt,
                      const bool                  computeVirial,
                      tensor                      virialScaled,
                      const PbcAiuc&              pbcAiuc)
{
    // Early exit if no settles
    if (numSettles_ == 0)
    {
        return;
    }

    if (computeVirial)
    {
        // Fill with zeros so the values can be reduced to it
        // Only 6 values are needed because virial is symmetrical
        clearDeviceBufferAsync(&d_virialScaled_, 0, 6, deviceStream_);
    }

    launchSettleGpuKernel(numSettles_,
                          d_atomIds_,
                          settleParameters_,
                          d_x,
                          d_xp,
                          updateVelocities,
                          d_v,
                          invdt,
                          computeVirial,
                          d_virialScaled_,
                          pbcAiuc,
                          deviceStream_);


    if (computeVirial)
    {
        copyFromDeviceBuffer(
                h_virialScaled_.data(), &d_virialScaled_, 0, 6, deviceStream_, GpuApiCallBehavior::Sync, nullptr);

        // Mapping [XX, XY, XZ, YY, YZ, ZZ] internal format to a tensor object
        virialScaled[XX][XX] += h_virialScaled_[0];
        virialScaled[XX][YY] += h_virialScaled_[1];
        virialScaled[XX][ZZ] += h_virialScaled_[2];

        virialScaled[YY][XX] += h_virialScaled_[1];
        virialScaled[YY][YY] += h_virialScaled_[3];
        virialScaled[YY][ZZ] += h_virialScaled_[4];

        virialScaled[ZZ][XX] += h_virialScaled_[2];
        virialScaled[ZZ][YY] += h_virialScaled_[4];
        virialScaled[ZZ][ZZ] += h_virialScaled_[5];
    }
}

SettleGpu::SettleGpu(const gmx_mtop_t& mtop, const DeviceContext& deviceContext, const DeviceStream& deviceStream) :
    deviceContext_(deviceContext), deviceStream_(deviceStream)
{
    static_assert(sizeof(real) == sizeof(float),
                  "Real numbers should be in single precision in GPU code.");

    // This is to prevent the assertion failure for the systems without water
    int totalSettles = computeTotalNumSettles(mtop);
    if (totalSettles == 0)
    {
        return;
    }

    auto settleParams = getSettleTopologyData(mtop);

    settleParameters_ = settleParameters(settleParams.mO,
                                         settleParams.mH,
                                         1.0 / settleParams.mO,
                                         1.0 / settleParams.mH,
                                         settleParams.dOH,
                                         settleParams.dHH);

    allocateDeviceBuffer(&d_virialScaled_, 6, deviceContext_);
    h_virialScaled_.resize(6);
}

SettleGpu::~SettleGpu()
{
    // Early exit if there is no settles
    if (numSettles_ == 0)
    {
        return;
    }
    // Wait for all the tasks to complete before freeing the memory. See #4519.
    deviceStream_.synchronize();

    freeDeviceBuffer(&d_virialScaled_);
    if (numAtomIdsAlloc_ > 0)
    {
        freeDeviceBuffer(&d_atomIds_);
    }
}

void SettleGpu::set(const InteractionDefinitions& idef)
{
    LocalSettleData localSettleData = computeNumSettles(idef);
    numSettles_                     = localSettleData.numSettle;
    int                 nral1       = localSettleData.nral;
    ArrayRef<const int> iatoms      = localSettleAtoms(idef);

    if (numSettles_)
    {
        reallocateDeviceBuffer(&d_atomIds_, numSettles_, &numAtomIds_, &numAtomIdsAlloc_, deviceContext_);
        h_atomIds_.resize(numSettles_);
        for (int i = 0; i < numSettles_; i++)
        {
            WaterMolecule settler;
            settler.ow1   = iatoms[i * nral1 + 1]; // Oxygen index
            settler.hw2   = iatoms[i * nral1 + 2]; // First hydrogen index
            settler.hw3   = iatoms[i * nral1 + 3]; // Second hydrogen index
            h_atomIds_[i] = settler;
        }
        copyToDeviceBuffer(
                &d_atomIds_, h_atomIds_.data(), 0, numSettles_, deviceStream_, GpuApiCallBehavior::Sync, nullptr);
    }
}

} // namespace gmx
