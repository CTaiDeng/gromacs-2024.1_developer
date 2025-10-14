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

#ifndef GMX_GMXPREPROCESS_TER_DB_H
#define GMX_GMXPREPROCESS_TER_DB_H

#include <filesystem>
#include <vector>

class PreprocessingAtomTypes;
struct MoleculePatchDatabase;

namespace gmx
{
template<typename>
class ArrayRef;
}

/*! \brief
 * Read database for N&C terminal modifications.
 *
 * \param[in] ffdir Directory for files.
 * \param[in] ter Which terminal side to read.
 * \param[inout] tbptr Database for terminii entry to populate.
 * \param[in] atype Database for atomtype information.
 * \returns Number of entries entered into database.
 */
int read_ter_db(const std::filesystem::path&        ffdir,
                char                                ter,
                std::vector<MoleculePatchDatabase>* tbptr,
                PreprocessingAtomTypes*             atype);

/*! \brief
 * Return entries for modification blocks that match a residue name.
 *
 * \param[in] tb Complete modification database.
 * \param[in] resname Residue name for terminus.
 * \returns A list of pointers to entries that match, or of nullptr for no matching entry.
 */
std::vector<MoleculePatchDatabase*> filter_ter(gmx::ArrayRef<MoleculePatchDatabase> tb, const char* resname);

/*! \brief
 * Interactively select one terminus.
 *
 * \param[in] tb List of possible entries, with pointer to actual entry or nullptr.
 * \param[in] title Name of entry.
 * \returns The modification block selected.
 */
MoleculePatchDatabase* choose_ter(gmx::ArrayRef<MoleculePatchDatabase*> tb, const char* title);

#endif
