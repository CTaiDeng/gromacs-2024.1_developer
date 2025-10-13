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

/* This file is completely threadsafe - keep it that way! */
#include "gmxpre.h"

#include "viewit.h"

#include <cstdlib>
#include <cstring>

#include <array>
#include <type_traits>

#include "gromacs/commandline/filenm.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"

static constexpr std::array<int, 5> canViewFileType = { 0, efEPS, efXPM, efXVG, efPDB };

static constexpr int numberOfPossibleFiles = canViewFileType.size();

static int can_view(int ftp)
{
    for (int i = 1; i < numberOfPossibleFiles; i++)
    {
        if (ftp == canViewFileType[i])
        {
            return i;
        }
    }

    return 0;
}

void do_view(const gmx_output_env_t* oenv, const char* fn, const char* opts)
{
    std::array<const char*, 5> viewProgram = {
        nullptr, "ghostview", "display", nullptr, "xterm -e rasmol"
    };
    char        buf[STRLEN], env[STRLEN];
    const char* cmd;
    int         ftp, n;

    if (output_env_get_view(oenv) && fn)
    {
        if (getenv("DISPLAY") == nullptr)
        {
            fprintf(stderr, "Can not view %s, no DISPLAY environment variable.\n", fn);
        }
        else
        {
            ftp = fn2ftp(fn);
            sprintf(env, "GMX_VIEW_%s", ftp2ext(ftp));
            upstring(env);
            switch (ftp)
            {
                case efXVG:
                    if (!(cmd = getenv(env)))
                    {
                        if (getenv("GMX_USE_XMGR"))
                        {
                            cmd = "xmgr";
                        }
                        else
                        {
                            cmd = "xmgrace";
                        }
                    }
                    break;
                default:
                    if ((n = can_view(ftp)))
                    {
                        if (!(cmd = getenv(env)))
                        {
                            cmd = viewProgram[n];
                        }
                    }
                    else
                    {
                        fprintf(stderr, "Don't know how to view file %s", fn);
                        return;
                    }
            }
            if (strlen(cmd))
            {
                sprintf(buf, "%s %s %s &", cmd, opts ? opts : "", fn);
                fprintf(stderr, "Executing '%s'\n", buf);
                if (0 != system(buf))
                {
                    gmx_fatal(FARGS, "Failed executing command: %s", buf);
                }
            }
        }
    }
}

void view_all(const gmx_output_env_t* oenv, int nf, t_filenm fnm[])
{
    int i;

    for (i = 0; i < nf; i++)
    {
        if (can_view(fnm[i].ftp) && is_output(&(fnm[i])) && (!is_optional(&(fnm[i])) || is_set(&(fnm[i]))))
        {
            do_view(oenv, fnm[i].filenames[0].c_str(), nullptr);
        }
    }
}
