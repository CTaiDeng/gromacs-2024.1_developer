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

#include "printtime.h"

#include "config.h"

#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/sysinfo.h"

void print_time(FILE*                     out,
                gmx_walltime_accounting_t walltime_accounting,
                int64_t                   step,
                const t_inputrec*         ir,
                const t_commrec*          cr)
{
    time_t finish;
    double dt, elapsed_seconds, time_per_step;

#if !GMX_THREAD_MPI
    if (!PAR(cr))
#endif
    {
        fprintf(out, "\r");
    }
    fputs("step ", out);
    fputs(gmx::int64ToString(step).c_str(), out);
    fflush(out);

    if ((step >= ir->nstlist))
    {
        double seconds_since_epoch = gmx_gettime();
        elapsed_seconds =
                seconds_since_epoch - walltime_accounting_get_start_time_stamp(walltime_accounting);
        time_per_step = elapsed_seconds / (step - ir->init_step + 1);
        dt            = (ir->nsteps + ir->init_step - step) * time_per_step;

        if (ir->nsteps >= 0)
        {
            if (dt >= 300)
            {
                finish       = static_cast<time_t>(seconds_since_epoch + dt);
                auto timebuf = gmx_ctime_r(&finish);
                timebuf.erase(timebuf.find_first_of('\n'));
                fputs(", will finish ", out);
                fputs(timebuf.c_str(), out);
            }
            else
            {
                fprintf(out, ", remaining wall clock time: %5d s          ", static_cast<int>(dt));
            }
        }
        else
        {
            fprintf(out, " performance: %.1f ns/day    ", ir->delta_t / 1000 * 24 * 60 * 60 / time_per_step);
        }
    }
#if !GMX_THREAD_MPI
    if (PAR(cr))
    {
        fprintf(out, "\n");
    }
#else
    GMX_UNUSED_VALUE(cr);
#endif

    fflush(out);
}

void print_date_and_time(FILE* fplog, int nodeid, const char* title, double the_time)
{
    if (!fplog)
    {
        return;
    }

    time_t temp_time = static_cast<time_t>(the_time);

    auto timebuf = gmx_ctime_r(&temp_time);

    fprintf(fplog, "%s on rank %d %s\n", title, nodeid, timebuf.c_str());
}

void print_start(FILE* fplog, const t_commrec* cr, gmx_walltime_accounting_t walltime_accounting, const char* name)
{
    char buf[STRLEN];

    sprintf(buf, "Started %s", name);
    print_date_and_time(
            fplog, cr->nodeid, buf, walltime_accounting_get_start_time_stamp(walltime_accounting));
}
