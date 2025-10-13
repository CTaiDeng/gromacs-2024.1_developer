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

#ifndef GMX_LINEARALGEBRA_MATRIX_H
#define GMX_LINEARALGEBRA_MATRIX_H

#include <cstdio>

double** alloc_matrix(int n, int m);

void free_matrix(double** a);

void matrix_multiply(FILE* fp, int n, int m, double** x, double** y, double** z);

/* Return 0 if OK or row number where inversion failed otherwise. */
int matrix_invert(FILE* fp, int n, double** a);

double multi_regression(FILE* fp, int ny, double* y, int nx, double** xx, double* a0);
/* Perform a regression analysis to fit
 * y' = a0[0] xx[0] + a0[1] xx[1] ... + a0[nx-1] xx[nx-1]
 * with ny data points in each vector.
 * The coefficients are returned in vector a0.
 * The return value of the function is the chi2 value:
 * sum_{j=0}^{ny-1} (y[j] - y'[j])^2
 * If fp is not NULL debug information will be written to it.
 */

#endif
