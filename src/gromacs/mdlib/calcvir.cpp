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

/* This file is completely threadsafe - keep it that way! */
#include "gmxpre.h"

#include "calcvir.h"

#include "config.h" /* for GMX_MAX_OPENMP_THREADS */

#include <algorithm>

#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/gmxassert.h"

static void upd_vir(rvec vir, real dvx, real dvy, real dvz)
{
    vir[XX] -= 0.5 * dvx;
    vir[YY] -= 0.5 * dvy;
    vir[ZZ] -= 0.5 * dvz;
}

static void calc_x_times_f(int nxf, const rvec x[], const rvec f[], gmx_bool bScrewPBC, const matrix box, matrix x_times_f)
{
    clear_mat(x_times_f);

    for (int i = 0; i < nxf; i++)
    {
        for (int d = 0; d < DIM; d++)
        {
            for (int n = 0; n < DIM; n++)
            {
                x_times_f[d][n] += x[i][d] * f[i][n];
            }
        }

        if (bScrewPBC)
        {
            int isx = gmx::shiftIndexToXDim(i);
            /* We should correct all odd x-shifts, but the range of isx is -2 to 2 */
            if (isx == 1 || isx == -1)
            {
                for (int d = 0; d < DIM; d++)
                {
                    for (int n = 0; n < DIM; n++)
                    {
                        x_times_f[d][n] += box[d][d] * f[i][n];
                    }
                }
            }
        }
    }
}

void calc_vir(int nxf, const rvec x[], const rvec f[], tensor vir, bool bScrewPBC, const matrix box)
{
    matrix x_times_f;

    int nthreads = gmx_omp_nthreads_get_simple_rvec_task(ModuleMultiThread::Default, nxf * 9);

    GMX_ASSERT(nthreads >= 1, "Avoids uninitialized x_times_f (warning)");

    if (nthreads == 1)
    {
        calc_x_times_f(nxf, x, f, bScrewPBC, box, x_times_f);
    }
    else
    {
        /* Use a buffer on the stack for storing thread-local results.
         * We use 2 extra elements (=18 reals) per thread to separate thread
         * local data by at least a cache line. Element 0 is not used.
         */
        matrix xf_buf[GMX_OPENMP_MAX_THREADS * 3];

#pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int thread = 0; thread < nthreads; thread++)
        {
            int start = (nxf * thread) / nthreads;
            int end   = std::min(nxf * (thread + 1) / nthreads, nxf);

            calc_x_times_f(end - start,
                           x + start,
                           f + start,
                           bScrewPBC,
                           box,
                           thread == 0 ? x_times_f : xf_buf[thread * 3]);
        }

        for (int thread = 1; thread < nthreads; thread++)
        {
            m_add(x_times_f, xf_buf[thread * 3], x_times_f);
        }
    }

    for (int d = 0; d < DIM; d++)
    {
        upd_vir(vir[d], x_times_f[d][XX], x_times_f[d][YY], x_times_f[d][ZZ]);
    }
}

void f_calc_vir(int i0, int i1, const rvec x[], const rvec f[], tensor vir, const matrix box)
{
    calc_vir(i1 - i0, x + i0, f + i0, vir, FALSE, box);
}
