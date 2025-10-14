/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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

#ifndef GMX_TOPOLOGY_RESIDUETYPES_H
#define GMX_TOPOLOGY_RESIDUETYPES_H

#include <string>
#include <unordered_map>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/stringutil.h"

/*! \brief Convenience type aliases
 *
 * These are not as useful as strong types, but they will
 * help clarify usage to humans in some cases. */
//! \{
using ResidueName = std::string;
using ResidueType = std::string;
//! \}

/*! \brief Maps residue names to residue types
 *
 * The contents are typically loaded from share/top/residuetypes.dat
 * or similar file provided in the users's working directory.
 */
using ResidueTypeMap =
        std::unordered_map<ResidueName, ResidueType, std::hash<ResidueName>, gmx::EqualCaseInsensitive>;

/*! \brief
 * Add entry to ResidueTypeMap if unique.
 *
 * \param[in] residueTypeMap Map to which to add new name+type entry
 * \param[in] residueName    Name of new residue.
 * \param[in] residueType    Type of new residue.
 */
void addResidue(ResidueTypeMap* residueTypeMap, const ResidueName& residueName, const ResidueType& residueType);

/*! \brief Returns a ResidueTypeMap filled from a file
 *
 * The value of the parameter is typically "residuetypes.dat" which
 * treats that as a GROMACS library file, ie. loads it from the working
 * directory or from "share/top" corresponding to the sourced GMXRC.
 *
 * \param[in] residueTypesDatFilename Library file to read and from which to fill the returned map
 */
ResidueTypeMap residueTypeMapFromLibraryFile(const std::string& residueTypesDatFilename);

/*! \brief
 * Checks if the indicated \p residueName is of \p residueType.
 *
 * \param[in] residueTypeMap Map to search
 * \param[in] residueName    Residue that should be checked.
 * \param[in] residueType    Which ResidueType the residue should have.
 * \returns If the check was successful.
 */
bool namedResidueHasType(const ResidueTypeMap& residueTypeMap,
                         const ResidueName&    residueName,
                         const ResidueType&    residueType);

/*! \brief
 * Return the residue type if a residue with that name exists, or "Other"
 *
 * \param[in] residueTypeMap Map to search
 * \param[in] residueName    Name of the residue to search for.
 * \returns The residue type of any matching residue, or "Other"
 */
ResidueType typeOfNamedDatabaseResidue(const ResidueTypeMap& residueTypeMap, const ResidueName& residueName);

#endif
