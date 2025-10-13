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

#include <cmath>
#include <cstring>

#include "gromacs/commandline/pargs.h"
#include "gromacs/commandline/viewit.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/gmxana/nrama.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/smalloc.h"


static void plot_rama(FILE* out, t_xrama* xr)
{
    int  i;
    real phi, psi;

    for (i = 0; (i < xr->npp); i++)
    {
        phi = xr->dih[xr->pp[i].iphi].ang * gmx::c_rad2Deg;
        psi = xr->dih[xr->pp[i].ipsi].ang * gmx::c_rad2Deg;
        fprintf(out, "%g  %g  %s\n", phi, psi, xr->pp[i].label);
    }
}

int gmx_rama(int argc, char* argv[])
{
    const char* desc[] = {
        "[THISMODULE] selects the [GRK]phi[grk]/[GRK]psi[grk] dihedral combinations from ",
        "your topology file and computes these as a function of time.",
        "Using simple Unix tools such as [IT]grep[it] you can select out specific residues."
    };

    FILE*             out;
    t_xrama*          xr;
    gmx_output_env_t* oenv;
    t_filenm          fnm[] = { { efTRX, "-f", nullptr, ffREAD },
                       { efTPR, nullptr, nullptr, ffREAD },
                       { efXVG, nullptr, "rama", ffWRITE } };
#define NFILE asize(fnm)

    if (!parse_common_args(
                &argc, argv, PCA_CAN_VIEW | PCA_CAN_TIME, NFILE, fnm, 0, nullptr, asize(desc), desc, 0, nullptr, &oenv))
    {
        return 0;
    }


    snew(xr, 1);
    init_rama(oenv, ftp2fn(efTRX, NFILE, fnm), ftp2fn(efTPR, NFILE, fnm), xr, 3);

    out = xvgropen(ftp2fn(efXVG, NFILE, fnm), "Ramachandran Plot", "Phi", "Psi", oenv);
    xvgr_line_props(out, 0, elNone, ecFrank, oenv);
    xvgr_view(out, 0.2, 0.2, 0.8, 0.8, oenv);
    xvgr_world(out, -180, -180, 180, 180, oenv);
    if (output_env_get_print_xvgr_codes(oenv))
    {
        fprintf(out, "@    xaxis  tick on\n@    xaxis  tick major 60\n@    xaxis  tick minor 30\n");
        fprintf(out, "@    yaxis  tick on\n@    yaxis  tick major 60\n@    yaxis  tick minor 30\n");
        fprintf(out, "@ s0 symbol 2\n@ s0 symbol size 0.4\n@ s0 symbol fill 1\n");
    }
    do
    {
        plot_rama(out, xr);
    } while (new_data(xr));
    fprintf(stderr, "\n");
    xvgrclose(out);

    do_view(oenv, ftp2fn(efXVG, NFILE, fnm), nullptr);

    return 0;
}
