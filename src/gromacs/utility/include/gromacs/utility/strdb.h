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

/*! \libinternal \file
 * \brief
 * Declares C functions for reading files with a list of strings.
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_STRDB_H
#define GMX_UTILITY_STRDB_H

#include <cstdio>

#include "gromacs/utility/basedefinitions.h"

/*! \brief
 * Reads a line of at most n characters from *fp to line.
 *
 * Comment ';...' and leading spaces are removed, empty lines are skipped.
 * Return FALSE when eof.
 */
gmx_bool get_a_line(FILE* fp, char line[], int n);

/*! \brief
 * Read a header between '[' and ']' from line to header.
 *
 * Returns FALSE if no header is found.
 */
gmx_bool get_header(char line[], char header[]);

/*! \brief
 * Opens file db, or if non-existant file $GMXLIB/db and read strings.
 *
 * First line in the file needs to specify the number of strings following.
 * Returns the number of strings.
 */
int get_lines(const char* db, char*** strings);

/*! \brief
 * Searches an array of strings for key, return the index if found.
 *
 * Returns -1 if not found.
 */
int search_str(int nstr, char** str, char* key);

#endif
