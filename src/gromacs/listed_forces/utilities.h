/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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

/*! \internal
 * \file
 *
 * \brief This file declares inline utility functionality.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_listed_forces
 */
#ifndef GMX_LISTED_FORCES_UTILITIES_H
#define GMX_LISTED_FORCES_UTILITIES_H

#include "gromacs/topology/ifunc.h"

/*! \brief Return whether this is an interaction that actually
 * calculates a potential and works on multiple atoms (not e.g. a
 * connection or a position restraint).
 *
 * \todo This function could go away when idef is not a big bucket of
 * everything. */
static bool ftype_is_bonded_potential(int ftype)
{
    return ((interaction_function[ftype].flags & IF_BOND) != 0U)
           && !(ftype == F_CONNBONDS || ftype == F_POSRES || ftype == F_FBPOSRES);
}

#endif
