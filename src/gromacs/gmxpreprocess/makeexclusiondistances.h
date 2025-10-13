/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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

#ifndef GMX_GMXPREPROCESS_READ_CONFORMATION_H
#define GMX_GMXPREPROCESS_READ_CONFORMATION_H

#include <vector>

#include "gromacs/utility/real.h"

class AtomProperties;
struct t_atoms;

/*! \brief Allocate and fill an array of inter-atomic half distances
 *
 * These are either scaled VDW radii taken from vdwradii.dat, or the
 * default value. Used directly and indirectly by solvate and
 * insert-molecules for deciding whether molecules clash. The return
 * pointer should be freed by the caller. */
std::vector<real> makeExclusionDistances(const t_atoms* a, AtomProperties* aps, real defaultDistance, real scaleFactor);

#endif
