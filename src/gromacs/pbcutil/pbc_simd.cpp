/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * \brief This file defines a low-level function for SIMD PBC calculation.
 *
 * \author Berk Hess <hess@kth.se>
 *
 * \ingroup module_pbcutil
 */
#include "gmxpre.h"

#include "gromacs/pbcutil/pbc_simd.h"

#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/simd/simd.h"

using namespace gmx; // TODO: Remove when this file is moved into gmx namespace

void set_pbc_simd(const t_pbc gmx_unused* pbc, real gmx_unused* pbc_simd)
{
#if GMX_SIMD_HAVE_REAL
    if (pbc != nullptr && pbc->pbcType != PbcType::No)
    {
        rvec inv_box_diag = { 0, 0, 0 };

        for (int d = 0; d < pbc->ndim_ePBC; d++)
        {
            inv_box_diag[d] = 1.0 / pbc->box[d][d];
        }

        store(pbc_simd + 0 * GMX_SIMD_REAL_WIDTH, SimdReal(inv_box_diag[ZZ]));
        store(pbc_simd + 1 * GMX_SIMD_REAL_WIDTH, SimdReal(pbc->box[ZZ][XX]));
        store(pbc_simd + 2 * GMX_SIMD_REAL_WIDTH, SimdReal(pbc->box[ZZ][YY]));
        store(pbc_simd + 3 * GMX_SIMD_REAL_WIDTH, SimdReal(pbc->box[ZZ][ZZ]));
        store(pbc_simd + 4 * GMX_SIMD_REAL_WIDTH, SimdReal(inv_box_diag[YY]));
        store(pbc_simd + 5 * GMX_SIMD_REAL_WIDTH, SimdReal(pbc->box[YY][XX]));
        store(pbc_simd + 6 * GMX_SIMD_REAL_WIDTH, SimdReal(pbc->box[YY][YY]));
        store(pbc_simd + 7 * GMX_SIMD_REAL_WIDTH, SimdReal(inv_box_diag[XX]));
        store(pbc_simd + 8 * GMX_SIMD_REAL_WIDTH, SimdReal(pbc->box[XX][XX]));
    }
    else
    {
        /* Setting inv_box_diag to zero leads to no PBC being applied */
        for (int i = 0; i < (DIM + DIM * (DIM + 1) / 2); i++)
        {
            store(pbc_simd + i * GMX_SIMD_REAL_WIDTH, SimdReal(0));
        }
    }
#endif
}
