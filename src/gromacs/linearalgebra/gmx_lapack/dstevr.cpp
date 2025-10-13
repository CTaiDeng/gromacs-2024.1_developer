/*
 * Copyright (C) 2025- GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
#include "../gmx_lapack.h"


/* Normally, DSTEVR is the LAPACK wrapper which calls one
 * of the eigenvalue methods. However, our code includes a
 * version of DSTEGR which is never than LAPACK 3.0 and can
 * handle requests for a subset of eigenvalues/vectors too,
 * and it should not need to call DSTEIN.
 * Just in case somebody has a faster version in their lapack
 * library we still call the driver routine, but in our own
 * case this is just a wrapper to dstegr.
 */
void F77_FUNC(dstevr, DSTEVR)(const char* jobz,
                              const char* range,
                              int*        n,
                              double*     d,
                              double*     e,
                              double*     vl,
                              double*     vu,
                              int*        il,
                              int*        iu,
                              double*     abstol,
                              int*        m,
                              double*     w,
                              double*     z,
                              int*        ldz,
                              int*        isuppz,
                              double*     work,
                              int*        lwork,
                              int*        iwork,
                              int*        liwork,
                              int*        info)
{
    F77_FUNC(dstegr, DSTEGR)
    (jobz, range, n, d, e, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info);


    return;
}
