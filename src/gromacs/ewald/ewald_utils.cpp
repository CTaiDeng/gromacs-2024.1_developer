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

#include "gmxpre.h"

#include "ewald_utils.h"

#include <cmath>

#include "gromacs/math/utilities.h"
#include "gromacs/utility/real.h"

real calc_ewaldcoeff_q(real rc, real rtol)
{
    real beta = 5, low, high;
    int  n, i = 0;

    do
    {
        i++;
        beta *= 2;
    } while (std::erfc(beta * rc) > rtol);

    /* Do a binary search with tolerance 2^-60 */
    n    = i + 60;
    low  = 0;
    high = beta;
    for (i = 0; i < n; i++)
    {
        beta = (low + high) / 2;
        if (std::erfc(beta * rc) > rtol)
        {
            low = beta;
        }
        else
        {
            high = beta;
        }
    }
    return beta;
}

static real compute_lj_function(real beta, real rc)
{
    real xrc, xrc2, xrc4, result;
    xrc    = beta * rc;
    xrc2   = xrc * xrc;
    xrc4   = xrc2 * xrc2;
    result = std::exp(-xrc2) * (1 + xrc2 + xrc4 / 2.0);

    return result;
}

real calc_ewaldcoeff_lj(real rc, real rtol)
{
    real beta = 5, low, high;
    int  n, i = 0;

    do
    {
        i++;
        beta *= 2.0;
    } while (compute_lj_function(beta, rc) > rtol);

    /* Do a binary search with tolerance 2^-60 */
    n    = i + 60;
    low  = 0;
    high = beta;
    for (i = 0; i < n; ++i)
    {
        beta = (low + high) / 2.0;
        if (compute_lj_function(beta, rc) > rtol)
        {
            low = beta;
        }
        else
        {
            high = beta;
        }
    }
    return beta;
}
