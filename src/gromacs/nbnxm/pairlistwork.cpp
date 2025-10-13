/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2023- The GROMACS Authors
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
 * \brief
 * Implements constructors for NbnxnPairlistGpuWork
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#include "gmxpre.h"

#include "pairlistwork.h"

#include "gromacs/simd/simd.h"

#include "boundingboxes.h"

NbnxnPairlistGpuWork::ISuperClusterData::ISuperClusterData() :
    bb(c_gpuNumClusterPerCell),
#if NBNXN_SEARCH_BB_SIMD4
    bbPacked(c_gpuNumClusterPerCell / c_packedBoundingBoxesDimSize * c_packedBoundingBoxesSize),
#endif
    x(c_gpuNumClusterPerCell * c_nbnxnGpuClusterSize * DIM),
    xSimd(c_gpuNumClusterPerCell * c_nbnxnGpuClusterSize * DIM)
{
}
NbnxnPairlistGpuWork::NbnxnPairlistGpuWork() :
    distanceBuffer(c_gpuNumClusterPerCell), sci_sort({}, { gmx::PinningPolicy::PinnedIfSupported })
{
}
