/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#include "pme_spline_work.h"

#include "gromacs/simd/simd.h"
#include "gromacs/utility/alignedallocator.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"

#include "pme_simd.h"

using namespace gmx; // TODO: Remove when this file is moved into gmx namespace

pme_spline_work* make_pme_spline_work(int gmx_unused order)
{
    pme_spline_work* work;

#ifdef PME_SIMD4_SPREAD_GATHER
    alignas(GMX_SIMD_ALIGNMENT) real tmp[GMX_SIMD4_WIDTH * 2];
    Simd4Real                        zero_S;
    Simd4Real                        real_mask_S0, real_mask_S1;
    int                              of, i;

    work = new (gmx::AlignedAllocationPolicy::malloc(sizeof(pme_spline_work))) pme_spline_work;

    zero_S = setZero();

    /* Generate bit masks to mask out the unused grid entries,
     * as we only operate on order of the 8 grid entries that are
     * load into 2 SIMD registers.
     */
    for (of = 0; of < 2 * GMX_SIMD4_WIDTH - (order - 1); of++)
    {
        for (i = 0; i < 2 * GMX_SIMD4_WIDTH; i++)
        {
            tmp[i] = (i >= of && i < of + order ? -1.0 : 1.0);
        }
        real_mask_S0      = load4(tmp);
        real_mask_S1      = load4(tmp + GMX_SIMD4_WIDTH);
        work->mask_S0[of] = (real_mask_S0 < zero_S);
        work->mask_S1[of] = (real_mask_S1 < zero_S);
    }
#else
    work = nullptr;
#endif

    return work;
}

void destroy_pme_spline_work(pme_spline_work* work)
{
    if (work != nullptr)
    {
        gmx::AlignedAllocationPolicy::free(work);
    }
}
