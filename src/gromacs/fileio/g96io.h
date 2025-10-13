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

#ifndef GMX_FILEIO_G96IO_H
#define GMX_FILEIO_G96IO_H

#include <cstdio>

#include <filesystem>

struct t_symtab;
struct t_trxframe;

int read_g96_conf(FILE*                        fp,
                  const std::filesystem::path& infile,
                  char**                       name,
                  struct t_trxframe*           fr,
                  struct t_symtab*             symtab,
                  char*                        line);
/* read a Gromos96 coordinate or trajectory file,                       *
 * returns the number of atoms                                          *
 * sets what's in the frame in info                                     *
 * read from fp, infile is only needed for error messages               *
 * nwanted is the number of wanted coordinates,                         *
 * set this to -1 if you want to know the number of atoms in the file   *
 * title, atoms, x, v can all be NULL, in which case they won't be read *
 * line holds the previous line for trajectory reading                  *
 *
 * symtab only needs to be valid if fr->atoms is valid
 *
 * If name is not nullptr, gmx_strdup the first g96 title string into it. */

void write_g96_conf(FILE* out, const char* title, const t_trxframe* fr, int nindex, const int* index);
/* write a Gromos96 coordinate file or trajectory frame *
 * index can be NULL                                    */

#endif
