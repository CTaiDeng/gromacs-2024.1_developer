/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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

#include "gmxpre.h"

#include "nbnxm_geometry.h"

#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/real.h"

#include "pairlist.h"

/* Clusters at the cut-off only increase rlist by 60% of their size */
static constexpr real c_nbnxnRlistIncreaseOutsideFactor = 0.6;

real nbnxn_get_rlist_effective_inc(const int jClusterSize, const real atomDensity)
{
    /* We should get this from the setup, but currently it's the same for
     * all setups, including GPUs.
     */
    const real iClusterSize = c_nbnxnCpuIClusterSize;

    const real iVolumeIncrease = (iClusterSize - 1) / atomDensity;
    const real jVolumeIncrease = (jClusterSize - 1) / atomDensity;

    return c_nbnxnRlistIncreaseOutsideFactor * std::cbrt(iVolumeIncrease + jVolumeIncrease);
}

real nbnxn_get_rlist_effective_inc(const int clusterSize, const gmx::RVec& averageClusterBoundingBox)
{
    /* The average length of the diagonal of a sub cell */
    const real diagonal = std::sqrt(norm2(averageClusterBoundingBox));

    const real volumeRatio = (clusterSize - 1.0_real) / clusterSize;

    return c_nbnxnRlistIncreaseOutsideFactor * gmx::square(volumeRatio) * 0.5_real * diagonal;
}
