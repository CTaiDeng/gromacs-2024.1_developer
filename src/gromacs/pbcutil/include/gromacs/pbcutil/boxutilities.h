/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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

#ifndef GMX_PBCUTIL_BOXUTILITIES_H
#define GMX_PBCUTIL_BOXUTILITIES_H

#include <cstdio>

#include "gromacs/math/vectypes.h"

/*! \brief Change box components to preserve the relative box shape
 *
 * Typically used with bInit set to false, in which case it changes
 * box components to b[XX][XX]*box_rel to preserve the relative box
 * shape. If bInit is true, then the values in b are used to set
 * those in box_rel so that subsquent calls can use that box_rel to
 * adjust b to maintain a consistent box.
 */
void do_box_rel(int ndim, const matrix deform, matrix box_rel, matrix b, bool bInit);

namespace gmx
{

/*! \brief
 * Returns whether two boxes are of equal size and shape (within reasonable
 * tolerance).
 */
bool boxesAreEqual(const matrix box1, const matrix box2);

/*! \brief
 * Returns whether a box is only initialised to zero or not.
 */
bool boxIsZero(const matrix box);

} // namespace gmx

#endif
