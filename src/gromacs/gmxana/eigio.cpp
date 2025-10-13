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

#include "gmxpre.h"

#include "eigio.h"

#include "gromacs/fileio/trrio.h"
#include "gromacs/math/vec.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/smalloc.h"

void read_eigenvectors(const char* file,
                       int*        natoms,
                       gmx_bool*   bFit,
                       rvec**      xref,
                       gmx_bool*   bDMR,
                       rvec**      xav,
                       gmx_bool*   bDMA,
                       int*        nvec,
                       int**       eignr,
                       rvec***     eigvec,
                       real**      eigval)
{
    gmx_trr_header_t head;
    int              i, snew_size;
    struct t_fileio* status;
    rvec*            x;
    matrix           box;
    gmx_bool         bOK;

    *bDMR = FALSE;

    /* read (reference (t=-1) and) average (t=0) structure */
    status = gmx_trr_open(file, "r");
    gmx_trr_read_frame_header(status, &head, &bOK);
    *natoms = head.natoms;
    snew(*xav, *natoms);
    gmx_trr_read_frame_data(status, &head, box, *xav, nullptr, nullptr);

    if ((head.t >= -1.1) && (head.t <= -0.9))
    {
        snew(*xref, *natoms);
        for (i = 0; i < *natoms; i++)
        {
            copy_rvec((*xav)[i], (*xref)[i]);
        }
        *bDMR = (head.lambda > 0.5);
        *bFit = (head.lambda > -0.5);
        if (*bFit)
        {
            fprintf(stderr,
                    "Read %smass weighted reference structure with %d atoms from %s\n",
                    *bDMR ? "" : "non ",
                    *natoms,
                    file);
        }
        else
        {
            fprintf(stderr, "Eigenvectors in %s were determined without fitting\n", file);
            sfree(*xref);
            *xref = nullptr;
        }
        gmx_trr_read_frame_header(status, &head, &bOK);
        gmx_trr_read_frame_data(status, &head, box, *xav, nullptr, nullptr);
    }
    else
    {
        *bFit = TRUE;
        *xref = nullptr;
    }
    *bDMA = (head.lambda > 0.5);
    if ((head.t <= -0.01) || (head.t >= 0.01))
    {
        fprintf(stderr,
                "WARNING: %s does not start with t=0, which should be the "
                "average structure. This might not be a eigenvector file. "
                "Some things might go wrong.\n",
                file);
    }
    else
    {
        fprintf(stderr,
                "Read %smass weighted average/minimum structure with %d atoms from %s\n",
                *bDMA ? "" : "non ",
                *natoms,
                file);
    }

    snew(x, *natoms);
    snew_size = 10;
    snew(*eignr, snew_size);
    snew(*eigval, snew_size);
    snew(*eigvec, snew_size);

    *nvec = 0;
    while (gmx_trr_read_frame_header(status, &head, &bOK))
    {
        gmx_trr_read_frame_data(status, &head, box, x, nullptr, nullptr);
        if (*nvec >= snew_size)
        {
            snew_size += 10;
            srenew(*eignr, snew_size);
            srenew(*eigval, snew_size);
            srenew(*eigvec, snew_size);
        }
        i                = head.step;
        (*eigval)[*nvec] = head.t;
        (*eignr)[*nvec]  = i - 1;
        snew((*eigvec)[*nvec], *natoms);
        for (i = 0; i < *natoms; i++)
        {
            copy_rvec(x[i], (*eigvec)[*nvec][i]);
        }
        (*nvec)++;
    }
    sfree(x);
    gmx_trr_close(status);
    fprintf(stderr, "Read %d eigenvectors (for %d atoms)\n\n", *nvec, *natoms);
}


void write_eigenvectors(const char* trrname,
                        int         natoms,
                        const real  mat[],
                        gmx_bool    bReverse,
                        int         begin,
                        int         end,
                        int         WriteXref,
                        const rvec* xref,
                        gmx_bool    bDMR,
                        const rvec  xav[],
                        gmx_bool    bDMA,
                        const real  eigval[])
{
    struct t_fileio* trrout;
    int              ndim, i, j, d, vec;
    matrix           zerobox;
    rvec*            x;

    ndim = natoms * DIM;
    clear_mat(zerobox);
    snew(x, natoms);

    fprintf(stderr,
            "\nWriting %saverage structure & eigenvectors %d--%d to %s\n",
            (WriteXref == eWXR_YES) ? "reference, " : "",
            begin,
            end,
            trrname);

    trrout = gmx_trr_open(trrname, "w");
    if (WriteXref == eWXR_YES)
    {
        /* misuse lambda: 0/1 mass weighted fit no/yes */
        gmx_trr_write_frame(trrout, -1, -1, bDMR ? 1.0 : 0.0, zerobox, natoms, xref, nullptr, nullptr);
    }
    else if (WriteXref == eWXR_NOFIT)
    {
        /* misuse lambda: -1 no fit */
        gmx_trr_write_frame(trrout, -1, -1, -1.0, zerobox, natoms, x, nullptr, nullptr);
    }

    /* misuse lambda: 0/1 mass weighted analysis no/yes */
    gmx_trr_write_frame(trrout, 0, 0, bDMA ? 1.0 : 0.0, zerobox, natoms, xav, nullptr, nullptr);

    for (i = 0; i <= (end - begin); i++)
    {

        if (!bReverse)
        {
            vec = i;
        }
        else
        {
            vec = ndim - i - 1;
        }

        for (j = 0; j < natoms; j++)
        {
            for (d = 0; d < DIM; d++)
            {
                x[j][d] = mat[vec * ndim + DIM * j + d];
            }
        }

        /* Store the eigenvalue in the time field */
        gmx_trr_write_frame(trrout, begin + i, eigval[vec], 0, zerobox, natoms, x, nullptr, nullptr);
    }
    gmx_trr_close(trrout);

    sfree(x);
}
