/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/*! \internal \file
 *
 * \brief Declares utility functions used in the domain decomposition module.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */

#include "gmxpre.h"

#include "utility.h"

#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/utility/fatalerror.h"

#include "domdec_internal.h"

char dim2char(int dim)
{
    char c = '?';

    switch (dim)
    {
        case XX: c = 'X'; break;
        case YY: c = 'Y'; break;
        case ZZ: c = 'Z'; break;
        default: gmx_fatal(FARGS, "Unknown dim %d", dim);
    }

    return c;
}

void make_tric_corr_matrix(int npbcdim, const matrix box, matrix tcm)
{
    if (YY < npbcdim)
    {
        tcm[YY][XX] = -box[YY][XX] / box[YY][YY];
    }
    else
    {
        tcm[YY][XX] = 0;
    }
    if (ZZ < npbcdim)
    {
        tcm[ZZ][XX] = -(box[ZZ][YY] * tcm[YY][XX] + box[ZZ][XX]) / box[ZZ][ZZ];
        tcm[ZZ][YY] = -box[ZZ][YY] / box[ZZ][ZZ];
    }
    else
    {
        tcm[ZZ][XX] = 0;
        tcm[ZZ][YY] = 0;
    }
}

void check_screw_box(const matrix box)
{
    /* Mathematical limitation */
    if (box[YY][XX] != 0 || box[ZZ][XX] != 0)
    {
        gmx_fatal(FARGS,
                  "With screw pbc the unit cell can not have non-zero off-diagonal x-components");
    }

    /* Limitation due to the asymmetry of the eighth shell method */
    if (box[ZZ][YY] != 0)
    {
        gmx_fatal(FARGS, "pbc=screw with non-zero box_zy is not supported");
    }
}

void dd_resize_atominfo_and_state(t_forcerec* fr, t_state* state, const int numAtoms)
{
    fr->atomInfo.resize(numAtoms);

    /* We use x during the setup of the atom communication */
    state->changeNumAtoms(numAtoms);
}
