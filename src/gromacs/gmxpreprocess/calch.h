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

#ifndef GMX_PREPROCESS_CALCH_H
#define GMX_PREPROCESS_CALCH_H

#include "gromacs/math/vectypes.h"

void calc_h_pos(int nht, rvec xa[], rvec xh[], int* l);
/*
 *    w.f. van gunsteren, groningen, july 1981
 *
 *    translated to c d. van der spoel groningen jun 1993
 *    added option 5 jan 95
 *
 *    subroutine genh (nht,nh,na,d,alfa,x)
 *
 *    genh generates cartesian coordinates for hydrogen atoms
 *    using the coordinates of neighbour atoms.
 *
 *    nht      : type of hydrogen attachment (see manual)
 *    xh(1.. ) : atomic positions of the hydrogen atoms that are to be
 *               generated
 *    xa(1..4) : atomic positions of the control atoms i, j and k and l
 *    default bond lengths and angles are defined internally
 *
 *    l : dynamically changed index
 */

#endif
