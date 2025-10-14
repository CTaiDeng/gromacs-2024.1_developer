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

/*! \libinternal \file
 * \brief
 * Declares routines to invert 3x3 matrices
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_math
 * \inlibraryapi
 */
#ifndef GMX_MATH_INVERTMATRIX_H
#define GMX_MATH_INVERTMATRIX_H

#include "gromacs/math/matrix.h"
#include "gromacs/math/vec.h"
#include "gromacs/utility/basedefinitions.h"

namespace gmx
{

/*! \brief Invert a general 3x3 matrix in \c src, return in \c dest
 *
 * A fatal error occurs if the determinant is too small. \c src and
 * \c dest cannot be the same matrix.
 */
void invertMatrix(const matrix src, matrix dest);

} // namespace gmx

#endif
