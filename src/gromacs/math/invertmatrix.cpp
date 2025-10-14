/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Routines to invert 3x3 matrices
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_math
 */
#include "gmxpre.h"

#include "gromacs/math/invertmatrix.h"

#include <cmath>

#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

void invertMatrix(const matrix src, matrix dest)
{
    const real smallreal = 1.0e-24_real;
    const real largereal = 1.0e24_real;

    real determinant = det(src);
    real c           = 1.0_real / determinant;
    real fc          = std::fabs(c);

    if ((fc <= smallreal) || (fc >= largereal))
    {
        gmx_fatal(FARGS, "Can not invert matrix, determinant = %e", determinant);
    }
    GMX_ASSERT(dest != src, "Cannot do in-place inversion of matrix");

    dest[XX][XX] = c * (src[YY][YY] * src[ZZ][ZZ] - src[ZZ][YY] * src[YY][ZZ]);
    dest[XX][YY] = -c * (src[XX][YY] * src[ZZ][ZZ] - src[ZZ][YY] * src[XX][ZZ]);
    dest[XX][ZZ] = c * (src[XX][YY] * src[YY][ZZ] - src[YY][YY] * src[XX][ZZ]);
    dest[YY][XX] = -c * (src[YY][XX] * src[ZZ][ZZ] - src[ZZ][XX] * src[YY][ZZ]);
    dest[YY][YY] = c * (src[XX][XX] * src[ZZ][ZZ] - src[ZZ][XX] * src[XX][ZZ]);
    dest[YY][ZZ] = -c * (src[XX][XX] * src[YY][ZZ] - src[YY][XX] * src[XX][ZZ]);
    dest[ZZ][XX] = c * (src[YY][XX] * src[ZZ][YY] - src[ZZ][XX] * src[YY][YY]);
    dest[ZZ][YY] = -c * (src[XX][XX] * src[ZZ][YY] - src[ZZ][XX] * src[XX][YY]);
    dest[ZZ][ZZ] = c * (src[XX][XX] * src[YY][YY] - src[YY][XX] * src[XX][YY]);
}

} // namespace gmx
