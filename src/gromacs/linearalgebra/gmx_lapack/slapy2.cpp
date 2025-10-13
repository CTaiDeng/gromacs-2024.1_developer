/*
 * Copyright (C) 2025- GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
#include <cmath>

#include "../gmx_lapack.h"

#include "gromacs/utility/real.h"

float F77_FUNC(slapy2, SLAPY2)(float* x, float* y)
{
    float xabs, yabs;
    float w, z;

    xabs = std::abs(*x);
    yabs = std::abs(*y);

    if (xabs > yabs)
    {
        w = xabs;
        z = yabs;
    }
    else
    {
        w = yabs;
        z = xabs;
    }

    if (std::abs(z) < GMX_FLOAT_MIN)
        return w;
    else
    {
        z = z / w;
        return w * std::sqrt(1.0 + z * z);
    }
}
