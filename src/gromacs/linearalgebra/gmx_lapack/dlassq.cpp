/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
#include <cmath>

#include "../gmx_lapack.h"

#include "gromacs/utility/real.h"

void F77_FUNC(dlassq, DLASSQ)(int* n, double* x, int* incx, double* scale, double* sumsq)
{
    int    ix;
    double absxi, t;

    if (*n > 0)
    {
        for (ix = 0; ix <= (*n - 1) * (*incx); ix += *incx)
        {
            if (std::abs(x[ix]) > GMX_DOUBLE_MIN)
            {
                absxi = std::abs(x[ix]);
                if (*scale < absxi)
                {
                    t      = *scale / absxi;
                    t      = t * t;
                    *sumsq = 1.0 + (*sumsq) * t;
                    *scale = absxi;
                }
                else
                {
                    t = absxi / (*scale);
                    *sumsq += t * t;
                }
            }
        }
    }
    return;
}
