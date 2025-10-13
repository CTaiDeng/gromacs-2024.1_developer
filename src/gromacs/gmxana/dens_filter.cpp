/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2011- The GROMACS Authors
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

/* dens_filter.c
 * Routines for Filters and convolutions
 */

#include <cmath>

#include "gromacs/math/vec.h"
#include "gromacs/utility/smalloc.h"

#include "dens_filter.h"

bool convolution(int dataSize, real* x, int kernelSize, const real* kernel)
{
    int   i, j, k;
    real* out;
    snew(out, dataSize);
    /* check validity of params */
    if (!x || !kernel)
    {
        return false;
    }
    if (dataSize <= 0 || kernelSize <= 0)
    {
        return false;
    }

    /* start convolution from out[kernelSize-1] to out[dataSize-1] (last) */
    for (i = kernelSize - 1; i < dataSize; ++i)
    {
        for (j = i, k = 0; k < kernelSize; --j, ++k)
        {
            out[i] += x[j] * kernel[k];
        }
    }

    /* convolution from out[0] to out[kernelSize-2] */
    for (i = 0; i < kernelSize - 1; ++i)
    {
        for (j = i, k = 0; j >= 0; --j, ++k)
        {
            out[i] += x[j] * kernel[k];
        }
    }

    for (i = 0; i < dataSize; i++)
    {
        x[i] = out[i];
    }
    sfree(out);
    return true;
}

/* Assuming kernel is shorter than x */

bool periodic_convolution(int datasize, real* x, int kernelsize, const real* kernel)
{
    int   i, j, idx;
    real* filtered;

    if (!x || !kernel)
    {
        return false;
    }
    if (kernelsize <= 0 || datasize <= 0 || kernelsize > datasize)
    {
        return false;
    }

    snew(filtered, datasize);

    for (i = 0; (i < datasize); i++)
    {
        for (j = 0; (j < kernelsize); j++)
        {
            // add datasize in case i-j is <0
            idx = i - j + datasize;
            filtered[i] += kernel[j] * x[idx % datasize];
        }
    }
    for (i = 0; i < datasize; i++)
    {
        x[i] = filtered[i];
    }
    sfree(filtered);

    return true;
}


/* returns discrete gaussian kernel of size n in *out, where n=2k+1=3,5,7,9,11 and k=1,2,3 is the
 * order NO checks are performed
 */
void gausskernel(real* out, int n, real var)
{
    int  i, j     = 0, k;
    real arg, tot = 0;
    k = n / 2;

    for (i = -k; i <= k; i++)
    {
        arg             = (i * i) / (2 * var);
        tot += out[j++] = std::exp(-arg);
    }
    for (i = 0; i < j; i++)
    {
        out[i] /= tot;
    }
}
