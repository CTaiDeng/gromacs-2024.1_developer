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

/*! \libinternal
 * \file
 * \brief Declares wrapper functions for higher-level matrix functions
 *
 * \ingroup module_math
 */
#ifndef GMX_MATH_NRJAC_H
#define GMX_MATH_NRJAC_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

/* Diagonalizes a symmetric matrix
 *
 * \param[in,out] a           Input matrix a[0..n-1][0..n-1] must be symmetric, gets modified
 * \param[in]  numDimensions  Number of rows and columns
 * \param[out] eigenvalues    eigenvalues[0]..eigenvalues[n-1] are the eigenvalues of a
 * \param[out] eigenvectors   v[0..n-1][0..n-1] the eigenvectors: v[i][j] is component i of vector j
 * \param[out] numRotations   The number of jacobi rotations, can be nullptr
 */
void jacobi(double** a, int numDimensions, double* eigenvalues, double** eigenvectors, int* numRotations);

/* Like jacobi above, but specialized for n=3
 *
 * \param[in,out] a  The symmetric matrix to diagonalize, size 3, note that the contents gets modified
 * \param[out] eigenvalues  The eigenvalues, size 3
 * \param[out] eigenvectors The eigenvectors, size 3

 * Returns the number of jacobi rotations.
 */
int jacobi(gmx::ArrayRef<gmx::DVec> a, gmx::ArrayRef<double> eigenvalues, gmx::ArrayRef<gmx::DVec> eigenvectors);

int m_inv_gen(const real* m, int n, real* minv);
/* Produces minv, a generalized inverse of m, both stored as linear arrays.
 * Inversion is done via diagonalization,
 * eigenvalues smaller than 1e-6 times the average diagonal element
 * are assumed to be zero.
 * For zero eigenvalues 1/eigenvalue is set to zero for the inverse matrix.
 * Returns the number of zero eigenvalues.
 */

#endif
