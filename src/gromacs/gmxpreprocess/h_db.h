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

#ifndef GMX_GMXPREPROCESS_H_DB_H
#define GMX_GMXPREPROCESS_H_DB_H

#include <cstdio>

#include <filesystem>
#include <vector>

#include "gromacs/utility/arrayref.h"

struct MoleculePatch;
struct MoleculePatchDatabase;

/* functions for the h-database */

void read_ab(char* line, const std::filesystem::path& fn, MoleculePatch* ab);
/* Read one add block */

/*! \brief
 * Read the databse from hdb file(s).
 *
 * \param[in] ffdir Directory for files.
 * \param[inout] globalPatches The database for atom modifications to populate.
 * \returns The number of modifications stored.
 */
int read_h_db(const std::filesystem::path& ffdir, std::vector<MoleculePatchDatabase>* globalPatches);

void print_ab(FILE* out, const MoleculePatch& ab, const char* nname);
/* print one add block */

/*! \brief
 * Search for an entry.
 *
 * \param[in] globalPatches Database to search.
 * \param[in] key Name to search for.
 */
gmx::ArrayRef<const MoleculePatchDatabase>::iterator
search_h_db(gmx::ArrayRef<const MoleculePatchDatabase> globalPatches, const char* key);
/* Search for an entry in the database */

#endif
