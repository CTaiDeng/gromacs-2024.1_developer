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

#include "gromacs/math/veccompare.h"

#include <cmath>
#include <cstdio>

#include "gromacs/utility/compare.h"

void cmp_rvec(FILE* fp, const char* s, int index, const rvec i1, const rvec i2, real ftol, real abstol)
{
    if (!equal_real(i1[XX], i2[XX], ftol, abstol) || !equal_real(i1[YY], i2[YY], ftol, abstol)
        || !equal_real(i1[ZZ], i2[ZZ], ftol, abstol))
    {
        if (index != -1)
        {
            fprintf(fp,
                    "%s[%5d] (%12.5e %12.5e %12.5e) - (%12.5e %12.5e %12.5e)\n",
                    s,
                    index,
                    i1[XX],
                    i1[YY],
                    i1[ZZ],
                    i2[XX],
                    i2[YY],
                    i2[ZZ]);
        }
        else
        {
            fprintf(fp,
                    "%s (%12.5e %12.5e %12.5e) - (%12.5e %12.5e %12.5e)\n",
                    s,
                    i1[XX],
                    i1[YY],
                    i1[ZZ],
                    i2[XX],
                    i2[YY],
                    i2[ZZ]);
        }
    }
}

void cmp_ivec(FILE* fp, const char* s, int index, const ivec i1, const ivec i2)
{
    if ((i1[XX] != i2[XX]) || (i1[YY] != i2[YY]) || (i1[ZZ] != i2[ZZ]))
    {
        if (index != -1)
        {
            fprintf(fp,
                    "%s[%5d] (%8d,%8d,%8d - %8d,%8d,%8d)\n",
                    s,
                    index,
                    i1[XX],
                    i1[YY],
                    i1[ZZ],
                    i2[XX],
                    i2[YY],
                    i2[ZZ]);
        }
        else
        {
            fprintf(fp, "%s (%8d,%8d,%8d - %8d,%8d,%8d)\n", s, i1[XX], i1[YY], i1[ZZ], i2[XX], i2[YY], i2[ZZ]);
        }
    }
}

static void cmp_rvecs_rmstol(FILE* fp, const char* title, int n, const rvec x1[], const rvec x2[], real ftol, real abstol)
{
    int    i, m;
    double rms;

    /* For a vector you are usally not interested in a relative difference
     * on a component that is very small compared to the other components.
     * Therefore we do the relative comparision relative to the RMS component.
     */
    rms = 0.0;
    for (i = 0; (i < n); i++)
    {
        for (m = 0; m < DIM; m++)
        {
            rms += x1[i][m] * x1[i][m] + x2[i][m] * x2[i][m];
        }
    }
    rms = sqrt(rms / (2 * n * DIM));

    /* Convert the relative tolerance into an absolute tolerance */
    if (ftol * rms < abstol)
    {
        abstol = ftol * rms;
    }

    /* And now do the actual comparision */
    for (i = 0; (i < n); i++)
    {
        cmp_rvec(fp, title, i, x1[i], x2[i], 0.0, abstol);
    }
}

void cmp_rvecs(FILE* fp, const char* title, int n, const rvec x1[], const rvec x2[], gmx_bool bRMSD, real ftol, real abstol)
{
    int    i, m;
    double d, ssd;

    if (bRMSD)
    {
        ssd = 0;
        for (i = 0; (i < n); i++)
        {
            for (m = 0; m < DIM; m++)
            {
                d = x1[i][m] - x2[i][m];
                ssd += d * d;
            }
        }
        fprintf(fp, "%s RMSD %g\n", title, std::sqrt(ssd / n));
    }
    else
    {
        cmp_rvecs_rmstol(fp, title, n, x1, x2, ftol, abstol);
    }
}
