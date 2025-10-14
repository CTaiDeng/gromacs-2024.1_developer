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

#ifndef GMX_FILEIO_GROIO_H
#define GMX_FILEIO_GROIO_H

#include <cstdio>

#include <filesystem>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"

struct gmx_mtop_t;
struct t_atoms;
struct t_symtab;
struct t_trxframe;

void get_coordnum(const std::filesystem::path& infile, int* natoms);
void gmx_gro_read_conf(const std::filesystem::path& infile,
                       t_symtab*                    symtab,
                       char**                       name,
                       t_atoms*                     atoms,
                       rvec                         x[],
                       rvec*                        v,
                       matrix                       box);
/* If name is not nullptr, gmx_strdup the title string into it. */

gmx_bool gro_next_x_or_v(FILE* status, struct t_trxframe* fr);
int      gro_first_x_or_v(FILE* status, struct t_trxframe* fr);
/* read first/next x and/or v frame from gro file */

void write_hconf_indexed_p(FILE*          out,
                           const char*    title,
                           const t_atoms* atoms,
                           int            nx,
                           const int      index[],
                           const rvec*    x,
                           const rvec*    v,
                           const matrix   box);

void write_hconf_mtop(FILE* out, const char* title, const gmx_mtop_t& mtop, const rvec* x, const rvec* v, const matrix box);

void write_hconf_p(FILE* out, const char* title, const t_atoms* atoms, const rvec* x, const rvec* v, const matrix box);
/* Write a Gromos file with precision ndec: number of decimal places in x,
 * v has one place more. */

void write_conf_p(const std::filesystem::path& outfile,
                  const char*                  title,
                  const t_atoms*               atoms,
                  const rvec*                  x,
                  const rvec*                  v,
                  const matrix                 box);

#endif
