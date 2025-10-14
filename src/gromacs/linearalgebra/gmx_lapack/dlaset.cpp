/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
#include <cctype>

#include "../gmx_lapack.h"


void F77_FUNC(dlaset,
              DLASET)(const char* uplo, int* m, int* n, double* alpha, double* beta, double* a, int* lda)
{
    int        i, j, k;
    const char ch = std::toupper(*uplo);

    if (ch == 'U')
    {
        for (j = 1; j < *n; j++)
        {
            k = (j < *m) ? j : *m;
            for (i = 0; i < k; i++)
                a[j * (*lda) + i] = *alpha;
        }
    }
    else if (ch == 'L')
    {
        k = (*m < *n) ? *m : *n;
        for (j = 0; j < k; j++)
        {
            for (i = j + 1; i < *m; i++)
                a[j * (*lda) + i] = *alpha;
        }
    }
    else
    {
        for (j = 0; j < *n; j++)
        {
            for (i = 0; i < *m; i++)
                a[j * (*lda) + i] = *alpha;
        }
    }

    k = (*m < *n) ? *m : *n;
    for (i = 0; i < k; i++)
        a[i * (*lda) + i] = *beta;
}
