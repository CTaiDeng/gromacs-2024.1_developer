/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#include "gmxpre.h"

#include "gromacs/topology/forcefieldparameters.h"

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/txtdump.h"

static void pr_cmap(FILE* fp, int indent, const char* title, const gmx_cmap_t* cmap_grid, gmx_bool bShowNumbers)
{
    const real dx = cmap_grid->grid_spacing != 0 ? 360.0 / cmap_grid->grid_spacing : 0;

    const int nelem = cmap_grid->grid_spacing * cmap_grid->grid_spacing;

    if (available(fp, cmap_grid, indent, title))
    {
        fprintf(fp, "%s\n", title);

        for (gmx::Index i = 0; i < gmx::ssize(cmap_grid->cmapdata); i++)
        {
            real idx = -180.0;
            fprintf(fp, "%8s %8s %8s %8s\n", "V", "dVdx", "dVdy", "d2dV");

            fprintf(fp, "grid[%3zd]={\n", bShowNumbers ? i : -1);

            for (int j = 0; j < nelem; j++)
            {
                if ((j % cmap_grid->grid_spacing) == 0)
                {
                    fprintf(fp, "%8.1f\n", idx);
                    idx += dx;
                }

                fprintf(fp, "%8.3f ", cmap_grid->cmapdata[i].cmap[j * 4]);
                fprintf(fp, "%8.3f ", cmap_grid->cmapdata[i].cmap[j * 4 + 1]);
                fprintf(fp, "%8.3f ", cmap_grid->cmapdata[i].cmap[j * 4 + 2]);
                fprintf(fp, "%8.3f\n", cmap_grid->cmapdata[i].cmap[j * 4 + 3]);
            }
            fprintf(fp, "\n");
        }
    }
}

void pr_ffparams(FILE* fp, int indent, const char* title, const gmx_ffparams_t* ffparams, gmx_bool bShowNumbers)
{
    indent = pr_title(fp, indent, title);
    pr_indent(fp, indent);
    fprintf(fp, "atnr=%d\n", ffparams->atnr);
    pr_indent(fp, indent);
    fprintf(fp, "ntypes=%d\n", ffparams->numTypes());
    for (int i = 0; i < ffparams->numTypes(); i++)
    {
        pr_indent(fp, indent + INDENT);
        fprintf(fp,
                "functype[%d]=%s, ",
                bShowNumbers ? i : -1,
                interaction_function[ffparams->functype[i]].name);
        pr_iparams(fp, ffparams->functype[i], ffparams->iparams[i]);
    }
    pr_double(fp, indent, "reppow", ffparams->reppow);
    pr_real(fp, indent, "fudgeQQ", ffparams->fudgeQQ);
    pr_cmap(fp, indent, "cmap", &ffparams->cmap_grid, bShowNumbers);
}
