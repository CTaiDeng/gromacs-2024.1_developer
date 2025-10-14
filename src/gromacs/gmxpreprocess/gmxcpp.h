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

#ifndef GMX_GMXPREPROCESS_GMXCPP_H
#define GMX_GMXPREPROCESS_GMXCPP_H

#include <filesystem>
#include <string>

typedef struct gmx_cpp* gmx_cpp_t;

/* The possible return codes for these functions */
enum
{
    eCPP_OK,
    eCPP_FILE_NOT_FOUND,
    eCPP_EOF,
    eCPP_SYNTAX,
    eCPP_INTERRUPT,
    eCPP_INVALID_HANDLE,
    eCPP_INVALID_INCLUDE_DELIMITER,
    eCPP_FILE_NOT_OPEN,
    eCPP_UNKNOWN,
    eCPP_NR
};

/* Open the file to be processed. The handle variable holds internal
   info for the cpp emulator. The cppopt variable (null terminated)
   can hold cpp options like -IXXX and -DXXX. Return integer status.
 */
int cpp_open_file(const std::filesystem::path& filenm, gmx_cpp_t* handlep, char** cppopts);

/* Return one whole line from the file into buf which holds at most n
   characters, for subsequent processing. Returns integer status.
 */
int cpp_read_line(gmx_cpp_t* handlep, int n, char buf[]);

/* Return the file currently being read.
 */
std::filesystem::path cpp_cur_file(const gmx_cpp_t* handlep);

/* Return the current line number.
 */
int cpp_cur_linenr(const gmx_cpp_t* handlep);

/* Close the file! Return integer status.
 */
int cpp_close_file(gmx_cpp_t* handlep);

/* Return a pointer to the value of defineName, when present, nullptr othwerwise.
 */
const std::string* cpp_find_define(const gmx_cpp_t* handlep, const std::string& defineName);

/* Clean up normal and file static data structures
 */
void cpp_done(gmx_cpp_t handle);

/* Return a string containing the error message coresponding to status
   variable.
 */
char* cpp_error(gmx_cpp_t* handlep, int status);

/* Returns warning message if strings defined in mdp define section (e.g. -DFLEXIBLE)
 * were not not found when processing the topology */
std::string checkAndWarnForUnusedDefines(const gmx_cpp& handle);

#endif
