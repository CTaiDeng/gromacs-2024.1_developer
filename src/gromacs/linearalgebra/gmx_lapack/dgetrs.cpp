/*
 * Copyright (C) 2025- GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
#include "../gmx_blas.h"
#include "../gmx_lapack.h"

void F77_FUNC(dgetrs,
              DGETRS)(const char* trans, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info)
{
    int    a_dim1, a_offset, b_dim1, b_offset;
    int    notran;
    int    c__1 = 1;
    int    c_n1 = -1;
    double one  = 1.0;

    a_dim1   = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    b_dim1   = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    *info  = 0;
    notran = (*trans == 'N' || *trans == 'n');

    if (*n <= 0 || *nrhs <= 0)
        return;

    if (notran)
    {
        F77_FUNC(dlaswp, DLASWP)(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c__1);
        F77_FUNC(dtrsm, DTRSM)
        ("Left", "Lower", "No transpose", "Unit", n, nrhs, &one, &a[a_offset], lda, &b[b_offset], ldb);

        F77_FUNC(dtrsm, DTRSM)
        ("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &one, &a[a_offset], lda, &b[b_offset], ldb);
    }
    else
    {
        F77_FUNC(dtrsm, DTRSM)
        ("Left", "Upper", "Transpose", "Non-unit", n, nrhs, &one, &a[a_offset], lda, &b[b_offset], ldb);
        F77_FUNC(dtrsm, DTRSM)
        ("Left", "Lower", "Transpose", "Unit", n, nrhs, &one, &a[a_offset], lda, &b[b_offset], ldb);

        F77_FUNC(dlaswp, DLASWP)(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c_n1);
    }

    return;
}
