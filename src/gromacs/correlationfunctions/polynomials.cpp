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

/*! \internal \file
 * \brief
 * Implements help function to compute Legendre polynomials
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \author Anders G&auml;rden&auml;s <anders.gardenas@gmail.com>
 * \ingroup module_correlationfunctions
 */
#include "gmxpre.h"

#include "polynomials.h"

#include "gromacs/utility/fatalerror.h"

real LegendreP(real x, unsigned int m)

{
    real polynomial = 0, x2, x3;

    switch (m)
    {
        case 0: polynomial = 1.0; break;
        case 1: polynomial = x; break;
        case 2:
            x2         = x * x;
            polynomial = 1.5 * x2 - 0.5;
            break;
        case 3:
            x2         = x * x;
            polynomial = (5 * x2 * x - 3 * x) * 0.5;
            break;
        case 4:
            x2         = x * x;
            polynomial = (35 * x2 * x2 - 30 * x2 + 3) / 8;
            break;
        case 5:
            x2         = x * x;
            x3         = x2 * x;
            polynomial = (63 * x3 * x2 - 70 * x3 + 15 * x) / 8;
            break;
        default: gmx_fatal(FARGS, "Legendre polynomials of order %u are not supported", m);
    }
    return (polynomial);
}
