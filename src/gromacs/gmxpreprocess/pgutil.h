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

#ifndef GMX_GMXPREPROCESS_PGUTIL_H
#define GMX_GMXPREPROCESS_PGUTIL_H

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct t_atom;
struct t_atoms;

/*! \brief
 * Search an atom in array of pointers to strings
 *
 * If \p type starts with '-', start from \p start.
 * Will search backwards from that.
 *
 * \p bondtype is only used for printing the error/warning string,
 * when \p bondtype ="check" no error/warning is issued.
 * When \p bAllowMissing = FALSE an fatal error is issued, otherwise a warning.
 *
 * \param[in] type             Atom type string to parse
 * \param[in] start            Possible position to begin search from.
 * \param[in] atoms            Global topology atoms information.
 * \param[in] bondtype         Information what kind of bond, used for error messages.
 * \param[in] bAllowMissing    If true, missing bond types are allowed.
 *                             AKA allow cartoon physics.
 * \param[in] cyclicBondsIndex Information about bonds creating cyclic molecules,
 *                             empty if no such bonds exist.
 */
int search_atom(const char*              type,
                int                      start,
                const t_atoms*           atoms,
                const char*              bondtype,
                bool                     bAllowMissing,
                gmx::ArrayRef<const int> cyclicBondsIndex);

/* Similar to search_atom, but this routine searches for the specified
 * atom in residue resind.
 */
int search_res_atom(const char* type, int resind, const t_atoms* atoms, const char* bondtype, bool bAllowMissing);

#endif
