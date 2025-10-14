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

#ifndef GMX_MATH_3DTRANSFORMS_H
#define GMX_MATH_3DTRANSFORMS_H

#include <cstdio>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"

/** Index for the fourth dimension for `vec4`. */
#define WW 3

/*! \brief
 * 4D vector type used in 3D transformations.
 *
 * In \Gromacs, only a limited set of 3D transformations are used, and all of
 * them operate on coordinates, so the fourth element is assumed to be one and
 * ignored in all contexts.
 */
typedef real vec4[4];

/*! \brief
 * 4D matrix type used in 3D transformations.
 */
typedef real mat4[4][4];

void gmx_mat4_copy(mat4 a, mat4 b);

void gmx_mat4_transform_point(mat4 m, const rvec x, vec4 v);

/*! \brief
 * Computes the product of two `mat4` matrices as A = B * C.
 *
 * Note that the order of operands is different from mmul() in vec.h!
 */
void gmx_mat4_mmul(mat4 A, mat4 B, mat4 C);

void gmx_mat4_init_unity(mat4 m);

void gmx_mat4_init_rotation(int axis, real angle, mat4 A);

void gmx_mat4_init_translation(real tx, real ty, real tz, mat4 A);

void gmx_mat4_print(FILE* fp, const char* s, mat4 A);

void gmx_vec4_print(FILE* fp, const char* s, vec4 a);

#endif
