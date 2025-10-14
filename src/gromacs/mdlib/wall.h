/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

#ifndef GMX_MDLIB_WALLS_H
#define GMX_MDLIB_WALLS_H

#include <cstdio>

#include "gromacs/math/vectypes.h"

struct SimulationGroups;
struct t_forcerec;
struct t_inputrec;
struct t_mdatoms;
struct t_nrnb;

namespace gmx
{
template<typename>
class ArrayRef;
class ForceWithVirial;
} // namespace gmx

void make_wall_tables(FILE*                   fplog,
                      const t_inputrec&       ir,
                      const char*             tabfn,
                      const SimulationGroups* groups,
                      t_forcerec*             fr);

real do_walls(const t_inputrec&                   ir,
              const t_forcerec&                   fr,
              const matrix                        box,
              gmx::ArrayRef<const int>            typeA,
              gmx::ArrayRef<const int>            typeB,
              gmx::ArrayRef<const unsigned short> cENER,
              int                                 homenr,
              int                                 numPerturbedAtoms,
              gmx::ArrayRef<const gmx::RVec>      x,
              gmx::ForceWithVirial*               forceWithVirial,
              real                                lambda,
              gmx::ArrayRef<real>                 Vlj,
              t_nrnb*                             nrnb);

#endif
