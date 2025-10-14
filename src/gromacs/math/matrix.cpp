/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 *
 * \brief Implements routines in matrix.h .
 *
 * \author Christian Blau <blau@kth.se>
 */

#include "gmxpre.h"

#include "gromacs/math/matrix.h"

namespace gmx
{


Matrix3x3 transpose(Matrix3x3ConstSpan matrixView)
{

    return Matrix3x3({ matrixView(0, 0),
                       matrixView(1, 0),
                       matrixView(2, 0),
                       matrixView(0, 1),
                       matrixView(1, 1),
                       matrixView(2, 1),
                       matrixView(0, 2),
                       matrixView(1, 2),
                       matrixView(2, 2) });
}

void matrixVectorMultiply(Matrix3x3ConstSpan matrix, RVec* v)
{
    const real resultXX =
            matrix(XX, XX) * (*v)[XX] + matrix(XX, YY) * (*v)[YY] + matrix(XX, ZZ) * (*v)[ZZ];
    const real resultYY =
            matrix(YY, XX) * (*v)[XX] + matrix(YY, YY) * (*v)[YY] + matrix(YY, ZZ) * (*v)[ZZ];
    (*v)[ZZ] = matrix(ZZ, XX) * (*v)[XX] + matrix(ZZ, YY) * (*v)[YY] + matrix(ZZ, ZZ) * (*v)[ZZ];
    (*v)[XX] = resultXX;
    (*v)[YY] = resultYY;
}


} // namespace gmx
