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

#include "rbin.h"

#include "gromacs/gmxlib/network.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/smalloc.h"

t_bin* mk_bin()
{
    t_bin* b;

    snew(b, 1);

    return b;
}

void destroy_bin(t_bin* b)
{
    if (b->maxreal > 0)
    {
        sfree(b->rbuf);
    }

    sfree(b);
}

void reset_bin(t_bin* b)
{
    b->nreal = 0;
}

int add_binr(t_bin* b, int nr, const real r[])
{
#define MULT 4
    int     i, rest, index;
    double* rbuf;

    if (b->nreal + nr > b->maxreal)
    {
        b->maxreal = b->nreal + nr;
        rest       = b->maxreal % MULT;
        if (rest != 0)
        {
            b->maxreal += MULT - rest;
        }
        srenew(b->rbuf, b->maxreal);
    }
    /* Copy pointer */
    rbuf = b->rbuf + b->nreal;

    for (i = 0; (i < nr); i++)
    {
        rbuf[i] = r[i];
    }

    index = b->nreal;
    b->nreal += nr;

    return index;
}

int add_binr(t_bin* b, gmx::ArrayRef<const real> r)
{
    return add_binr(b, r.size(), r.data());
}

int add_bind(t_bin* b, int nr, const double r[])
{
#define MULT 4
    int     i, rest, index;
    double* rbuf;

    if (b->nreal + nr > b->maxreal)
    {
        b->maxreal = b->nreal + nr;
        rest       = b->maxreal % MULT;
        if (rest != 0)
        {
            b->maxreal += MULT - rest;
        }
        srenew(b->rbuf, b->maxreal);
    }
    /* Copy pointer */
    rbuf = b->rbuf + b->nreal;
    for (i = 0; (i < nr); i++)
    {
        rbuf[i] = r[i];
    }

    index = b->nreal;
    b->nreal += nr;

    return index;
}

int add_bind(t_bin* b, gmx::ArrayRef<const double> r)
{
    return add_bind(b, r.size(), r.data());
}

void sum_bin(t_bin* b, const t_commrec* cr)
{
    int i;

    for (i = b->nreal; (i < b->maxreal); i++)
    {
        b->rbuf[i] = 0;
    }
    gmx_sumd(b->maxreal, b->rbuf, cr);
}

void extract_binr(t_bin* b, int index, int nr, real r[])
{
    int     i;
    double* rbuf;

    rbuf = b->rbuf + index;
    for (i = 0; (i < nr); i++)
    {
        r[i] = rbuf[i];
    }
}

void extract_binr(t_bin* b, int index, gmx::ArrayRef<real> r)
{
    extract_binr(b, index, r.size(), r.data());
}

void extract_bind(t_bin* b, int index, int nr, double r[])
{
    int     i;
    double* rbuf;

    rbuf = b->rbuf + index;
    for (i = 0; (i < nr); i++)
    {
        r[i] = rbuf[i];
    }
}

void extract_bind(t_bin* b, int index, gmx::ArrayRef<double> r)
{
    extract_bind(b, index, r.size(), r.data());
}
