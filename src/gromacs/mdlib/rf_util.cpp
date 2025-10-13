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

#include "rf_util.h"

#include <cmath>

#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/pleasecite.h"

void calc_rffac(FILE* fplog, real eps_r, real eps_rf, real Rc, real* krf, real* crf)
{
    /* eps == 0 signals infinite dielectric */
    if (eps_rf == 0)
    {
        *krf = 1 / (2 * Rc * Rc * Rc);
    }
    else
    {
        *krf = (eps_rf - eps_r) / (2 * eps_rf + eps_r) / (Rc * Rc * Rc);
    }
    *crf = 1 / Rc + *krf * Rc * Rc;

    if (fplog)
    {
        fprintf(fplog,
                "%s:\n"
                "epsRF = %g, rc = %g, krf = %g, crf = %g, epsfac = %g\n",
                enumValueToString(CoulombInteractionType::RF),
                eps_rf,
                Rc,
                *krf,
                *crf,
                gmx::c_one4PiEps0 / eps_r);
        if (*krf > 0)
        {
            // Make sure we don't lose resolution in pow() by casting real arg to double
            real rmin = gmx::invcbrt(static_cast<double>(*krf * 2.0));
            fprintf(fplog, "The electrostatics potential has its minimum at r = %g\n", rmin);
        }
    }
}
