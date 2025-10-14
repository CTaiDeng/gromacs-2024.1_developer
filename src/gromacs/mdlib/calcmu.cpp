/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

/* This file is completely threadsafe - keep it that way! */
#include "gmxpre.h"

#include "calcmu.h"

#include <cstdio>
#include <cstdlib>

#include "gromacs/gmxlib/network.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/utility/arrayref.h"

void calc_mu(int                            start,
             int                            homenr,
             gmx::ArrayRef<const gmx::RVec> x,
             gmx::ArrayRef<const real>      q,
             gmx::ArrayRef<const real>      qB,
             bool                           havePerturbedCharges,
             dvec                           mu,
             dvec                           mu_B)
{
    int    end, m;
    double mu_x, mu_y, mu_z;

    end = start + homenr;

    mu_x = mu_y = mu_z = 0.0;
#pragma omp parallel for reduction(+: mu_x, mu_y, mu_z) schedule(static) \
    num_threads(gmx_omp_nthreads_get(ModuleMultiThread::Default))
    for (int i = start; i < end; i++)
    {
        // Trivial OpenMP region that cannot throw
        mu_x += q[i] * x[i][XX];
        mu_y += q[i] * x[i][YY];
        mu_z += q[i] * x[i][ZZ];
    }
    mu[XX] = mu_x;
    mu[YY] = mu_y;
    mu[ZZ] = mu_z;

    for (m = 0; (m < DIM); m++)
    {
        mu[m] *= gmx::c_enm2Debye;
    }

    if (havePerturbedCharges)
    {
        mu_x = mu_y = mu_z = 0.0;
#pragma omp parallel for reduction(+: mu_x, mu_y, mu_z) schedule(static) \
        num_threads(gmx_omp_nthreads_get(ModuleMultiThread::Default))
        for (int i = start; i < end; i++)
        {
            // Trivial OpenMP region that cannot throw
            mu_x += qB[i] * x[i][XX];
            mu_y += qB[i] * x[i][YY];
            mu_z += qB[i] * x[i][ZZ];
        }
        mu_B[XX] = mu_x * gmx::c_enm2Debye;
        mu_B[YY] = mu_y * gmx::c_enm2Debye;
        mu_B[ZZ] = mu_z * gmx::c_enm2Debye;
    }
    else
    {
        copy_dvec(mu, mu_B);
    }
}
