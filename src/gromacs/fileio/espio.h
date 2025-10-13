/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2005- The GROMACS Authors
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

#ifndef GMX_FILEIO_ESPIO_H
#define GMX_FILEIO_ESPIO_H

#include <cstdio>

#include <filesystem>

#include "gromacs/math/vectypes.h"

struct t_atoms;
struct t_symtab;

void gmx_espresso_read_conf(const std::filesystem::path& infile,
                            t_symtab*                    symtab,
                            char**                       name,
                            t_atoms*                     atoms,
                            rvec                         x[],
                            rvec*                        v,
                            matrix                       box);
/* If name is not nullptr, gmx_strdup the title string into
 * it. Reading a title from espresso format is not , so this will
 * always be an empty string. */

int get_espresso_coordnum(const std::filesystem::path& infile);

void write_espresso_conf_indexed(FILE*          out,
                                 const char*    title,
                                 const t_atoms* atoms,
                                 int            nx,
                                 const int*     index,
                                 const rvec*    x,
                                 const rvec*    v,
                                 const matrix   box);

#endif
