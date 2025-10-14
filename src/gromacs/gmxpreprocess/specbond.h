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

#ifndef GMX_GMXPREPROCESS_SPECBOND_H
#define GMX_GMXPREPROCESS_SPECBOND_H

#include <string>
#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/topology/symtab.h"

struct t_atoms;
struct SpecialBond;

struct DisulfideBond
{
    int         firstResidue = -1, secondResidue = -1;
    std::string firstAtom, secondAtom;
};

std::vector<DisulfideBond>
makeDisulfideBonds(t_atoms* pdba, t_symtab* symtab, rvec x[], bool bInteractive, bool bVerbose);

std::vector<SpecialBond> generateSpecialBonds();

#endif
