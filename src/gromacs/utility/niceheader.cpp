/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * \brief
 * Implements functions from niceheader.h.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/niceheader.h"

#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/sysinfo.h"
#include "gromacs/utility/textwriter.h"

namespace gmx
{

void niceHeader(TextWriter* writer, const char* fn, char commentChar)
{
    char userbuf[256];
    char hostbuf[256];

    /* Write a nice header for an output file */
    writer->writeLine(formatString("%c", commentChar));
    writer->writeLine(formatString("%c\tFile '%s' was generated", commentChar, fn ? fn : "unknown"));

    int uid = gmx_getuid();
    gmx_getusername(userbuf, 256);
    gmx_gethostname(hostbuf, 256);

    writer->writeLine(formatString("%c\tBy user: %s (%d)", commentChar, userbuf, uid));
    writer->writeLine(formatString("%c\tOn host: %s", commentChar, hostbuf));
    writer->writeLine(formatString("%c\tAt date: %s", commentChar, gmx_format_current_time().c_str()));
    writer->writeLine(formatString("%c", commentChar));
}

} // namespace gmx
