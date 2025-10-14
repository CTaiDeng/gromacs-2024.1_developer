/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
// -*- c++ -*-

// TODO These functions will be removed in the near future and replaced wtih
// LGPL-compatible functions

namespace NR_Jacobi {

  /// Numerical recipes diagonalization
  int jacobi(cvm::real a[4][4], cvm::real d[4], cvm::real v[4][4], int *nrot);

  /// Eigenvector sort
  int eigsrt(cvm::real d[4], cvm::real v[4][4]);

  /// Transpose the matrix
  int transpose(cvm::real v[4][4]);

}

