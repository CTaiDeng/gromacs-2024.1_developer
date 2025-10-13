/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 *  \brief Implements PME GPU gather in SYCL.
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 */

#include "gromacs/gpu_utils/syclutils.h"

#include "pme_gpu_types_host.h"
#include "pme_grid.h"

struct PmeGpuGridParams;
struct PmeGpuAtomParams;
struct PmeGpuDynamicParams;

template<int order, bool wrapX, bool wrapY, int numGrids, bool readGlobal, ThreadsPerAtom threadsPerAtom, int subGroupSize>
class PmeGatherKernel : public ISyclKernelFunctor
{
public:
    PmeGatherKernel();
    void setArg(size_t argIndex, void* arg) override;
    void launch(const KernelLaunchConfig& config, const DeviceStream& deviceStream) override;

private:
    PmeGpuGridParams*    gridParams_;
    PmeGpuAtomParams*    atomParams_;
    PmeGpuDynamicParams* dynamicParams_;
    void                 reset();
};
