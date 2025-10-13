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

#ifndef GMX_CONFORMATION_UTILITIES_H
#define GMX_CONFORMATION_UTILITIES_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

void rotate_conf(int natom, rvec* x, rvec* v, real alfa, real beta, real gamma);
/*rotate() rotates a configuration alfa degrees around the x_axis and beta degrees around the y_axis, *v can be NULL */

void make_new_box(int natoms, rvec* x, matrix box, const rvec box_space, gmx_bool bCenter);
/* Generates a box around a configuration, box_space is optional extra
 * space around it. If bCenter then coordinates will be centered in
 * the generated box
 */

#endif
