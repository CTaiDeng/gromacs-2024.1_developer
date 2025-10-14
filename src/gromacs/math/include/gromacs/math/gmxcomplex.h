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

#ifndef GMX_MATH_GMXCOMPLEX_H
#define GMX_MATH_GMXCOMPLEX_H

#include <cmath>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"

struct t_complex
{
    real re, im;
};

static t_complex rcmul(real r, t_complex c)
{
    t_complex d;

    d.re = r * c.re;
    d.im = r * c.im;

    return d;
}

static inline t_complex rcexp(real r)
{
    t_complex c;

    c.re = cos(r);
    c.im = sin(r);

    return c;
}


static inline t_complex cadd(t_complex a, t_complex b)
{
    t_complex c;

    c.re = a.re + b.re;
    c.im = a.im + b.im;

    return c;
}

static inline t_complex csub(t_complex a, t_complex b)
{
    t_complex c;

    c.re = a.re - b.re;
    c.im = a.im - b.im;

    return c;
}

static t_complex cmul(t_complex a, t_complex b)
{
    t_complex c;

    c.re = a.re * b.re - a.im * b.im;
    c.im = a.re * b.im + a.im * b.re;

    return c;
}

static t_complex conjugate(t_complex c)
{
    t_complex d;

    d.re = c.re;
    d.im = -c.im;

    return d;
}

static inline real cabs2(t_complex c)
{
    real abs2;
    abs2 = (c.re * c.re) + (c.im * c.im);

    return abs2;
}

static inline t_complex cdiv(t_complex teller, t_complex noemer)
{
    t_complex res, anoemer;

    anoemer = cmul(conjugate(noemer), noemer);
    res     = cmul(teller, conjugate(noemer));

    return rcmul(1.0 / anoemer.re, res);
}

inline bool operator==(const t_complex& lhs, const t_complex& rhs)
{
    return (lhs.re == rhs.re) && (lhs.im == rhs.im);
}
inline bool operator!=(const t_complex& lhs, const t_complex& rhs)
{
    return !(lhs == rhs);
}

#endif
