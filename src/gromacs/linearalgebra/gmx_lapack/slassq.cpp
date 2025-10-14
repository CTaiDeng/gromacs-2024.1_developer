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

void F77_FUNC(slassq, SLASSQ)(int* n, float* x, int* incx, float* scale, float* sumsq)
{
    int   ix;
    float absxi, t;

    if (*n > 0)
    {
        for (ix = 0; ix <= (*n - 1) * (*incx); ix += *incx)
        {
            if (std::abs(x[ix]) > GMX_FLOAT_MIN)
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
