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

#ifndef GMX_GMXANA_EIGIO_H
#define GMX_GMXANA_EIGIO_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"

enum
{
    eWXR_NO,
    eWXR_YES,
    eWXR_NOFIT
};

extern void read_eigenvectors(const char* file,
                              int*        natoms,
                              bool*       bFit,
                              rvec**      xref,
                              bool*       bDMR,
                              rvec**      xav,
                              bool*       bDMA,
                              int*        nvec,
                              int**       eignr,
                              rvec***     eigvec,
                              real**      eigval);
/* Read eigenvectors from file into eigvec, the eigenvector numbers   */
/* are stored in eignr.                                               */
/* When bFit=FALSE no fitting was used in the covariance analysis.    */
/* xref is the reference structure, can be NULL if not present.       */
/* bDMR indicates mass weighted fit.                                  */
/* xav is the average/minimum structure is written (t=0).             */
/* bDMA indicates mass weighted analysis/eigenvectors.                */

extern void write_eigenvectors(const char* trrname,
                               int         natoms,
                               const real  mat[],
                               bool        bReverse,
                               int         begin,
                               int         end,
                               int         WriteXref,
                               const rvec* xref,
                               bool        bDMR,
                               const rvec  xav[],
                               bool        bDMA,
                               const real* eigval);
/* Write eigenvectors in mat to a TRR file.                           */
/* The reference structure is written (t=-1) when WriteXref=eWXR_YES. */
/* When WriteXref==eWXR_NOFIT a zero frame is written (t=-1),         */
/* with lambda=-1.                                                    */
/* bDMR indicates mass weighted fit.                                  */
/* The average/minimum structure is written (t=0).                    */
/* bDMA indicates mass weighted analysis/eigenvectors.                */
/* eigenvectors with begin <= num <= end are written (num is base-1), */
/* the timestamp of eigenvector num is num.                           */
/* If bReverse==TRUE, num=1 is the last vector in mat.                */

#endif
