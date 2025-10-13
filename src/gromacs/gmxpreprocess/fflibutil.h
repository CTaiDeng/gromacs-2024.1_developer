/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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

#ifndef GMX_GMXPREPROCESS_FFLIBUTIL_H
#define GMX_GMXPREPROCESS_FFLIBUTIL_H

#include <cstdio>

#include <filesystem>
#include <string>
#include <vector>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/datafilefinder.h"

/*! \brief
 * Enumerates forcefields in the data directories.
 */
std::vector<gmx::DataFileInfo> fflib_enumerate_forcefields();

std::filesystem::path fflib_forcefield_dir_ext();
/* Returns the name of the force field directory extension */

std::filesystem::path fflib_forcefield_itp();
/* Returns the name of the main forcefield itp file */

std::filesystem::path fflib_forcefield_doc();
/* Returns the name of the forcefield documentation file */

std::filesystem::path fflib_filename_base(const std::filesystem::path& filename);
/* Return the base file name of filename in base,
 * i.e. remove path and extension, if present.
 */

std::vector<std::filesystem::path> fflib_search_file_end(const std::filesystem::path& ffdir,
                                                         const char*                  file_end,
                                                         bool                         bFatalError);
/* Search for files ending on file_end in the force field directory fflib.
 * fflib should be in the GROMACS lib.path.
 * Return the number of files and the file names in filenames.
 */

bool fflib_fexist(const std::filesystem::path& file);
/* Check if a file exists in the force field library */

FILE* fflib_open(const std::filesystem::path& file);
/* Open force field library file "file" for reading.
 * "file" should contain the whole path to the force field library,
 * either absolute or relative to the current dir.
 */

#endif
