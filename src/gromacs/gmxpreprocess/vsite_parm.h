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

#ifndef GMX_GMXPREPROCESS_VSITE_PARM_H
#define GMX_GMXPREPROCESS_VSITE_PARM_H

class PreprocessingAtomTypes;
struct gmx_moltype_t;
struct t_atoms;
struct InteractionsOfType;

namespace gmx
{
template<typename>
class ArrayRef;
class MDLogger;
} // namespace gmx

int set_vsites(bool                              bVerbose,
               t_atoms*                          atoms,
               PreprocessingAtomTypes*           atype,
               gmx::ArrayRef<InteractionsOfType> plist,
               const gmx::MDLogger&              logger);
/* set parameters for virtual sites, return number of virtual sites */

void set_vsites_ptype(bool bVerbose, gmx_moltype_t* molt, const gmx::MDLogger& logger);
/* set ptype to VSite for virtual sites */

/*! \brief Clean up the bonded interactions
 *
 * Throw away all obsolete bonds, angles and dihedrals.
 * Throw away all constraints. */
void clean_vsite_bondeds(gmx::ArrayRef<InteractionsOfType> ps,
                         int                               natoms,
                         bool                              bRmVSiteBds,
                         const gmx::MDLogger&              logger);

#endif
