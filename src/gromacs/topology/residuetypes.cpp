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

#include "gmxpre.h"

#include "gromacs/topology/residuetypes.h"

#include <string>

#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/strdb.h"

//! Definition for residue type that is not known.
const ResidueType c_undefinedResidueType = "Other";

void addResidue(ResidueTypeMap* residueTypeMap, const ResidueName& residueName, const ResidueType& residueType)
{
    if (auto [foundIt, insertionTookPlace] = residueTypeMap->insert({ residueName, residueType });
        !insertionTookPlace)
    {
        if (!gmx::equalCaseInsensitive(foundIt->second, residueType))
        {
            fprintf(stderr,
                    "Warning: Residue '%s' already present with type '%s' in database, ignoring "
                    "new type '%s'.\n",
                    residueName.c_str(),
                    foundIt->second.c_str(),
                    residueType.c_str());
        }
    }
}

ResidueTypeMap residueTypeMapFromLibraryFile(const std::string& residueTypesDatFilename)
{
    char line[STRLEN];
    char resname[STRLEN], restype[STRLEN], dum[STRLEN];

    gmx::FilePtr db = gmx::openLibraryFile(residueTypesDatFilename);

    ResidueTypeMap residueTypeMap;
    while (get_a_line(db.get(), line, STRLEN))
    {
        strip_comment(line);
        trim(line);
        if (line[0] != '\0')
        {
            if (sscanf(line, "%1000s %1000s %1000s", resname, restype, dum) != 2)
            {
                gmx_fatal(
                        FARGS,
                        "Incorrect number of columns (2 expected) for line in residuetypes.dat  ");
            }
            addResidue(&residueTypeMap, resname, restype);
        }
    }
    return residueTypeMap;
}

bool namedResidueHasType(const ResidueTypeMap& residueTypeMap,
                         const ResidueName&    residueName,
                         const ResidueType&    residueType)
{
    if (auto foundIt = residueTypeMap.find(residueName); foundIt != residueTypeMap.end())
    {
        return gmx::equalCaseInsensitive(residueType, foundIt->second);
    }
    return false;
}

ResidueType typeOfNamedDatabaseResidue(const ResidueTypeMap& residueTypeMap, const ResidueName& residueName)
{
    if (auto foundIt = residueTypeMap.find(residueName); foundIt != residueTypeMap.end())
    {
        return foundIt->second;
    }
    return c_undefinedResidueType;
}
