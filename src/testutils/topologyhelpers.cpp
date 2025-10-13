/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2022- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Helper functions for topology generation.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 */

#include "gmxpre.h"

#include "topologyhelpers.h"

#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"

namespace gmx
{
namespace test
{

void addNWaterMolecules(gmx_mtop_t* mtop, int numWaters)
{
    gmx_moltype_t moltype;
    moltype.atoms.nr             = NRAL(F_SETTLE);
    std::vector<int>& iatoms     = moltype.ilist[F_SETTLE].iatoms;
    const int         settleType = 0;
    iatoms.push_back(settleType);
    iatoms.push_back(0);
    iatoms.push_back(1);
    iatoms.push_back(2);
    int moleculeTypeIndex = mtop->moltype.size();
    mtop->moltype.push_back(moltype);
    init_t_atoms(&mtop->moltype[0].atoms, NRAL(F_SETTLE), false);
    for (int i = 0; i < NRAL(F_SETTLE); ++i)
    {
        mtop->moltype[0].atoms.atom[i].m = (i % 3 == 0) ? 16 : 1;
    }

    mtop->molblock.emplace_back(gmx_molblock_t{});
    mtop->molblock.back().type = moleculeTypeIndex;
    mtop->molblock.back().nmol = numWaters;
    mtop->natoms               = moltype.atoms.nr * mtop->molblock.back().nmol;
}

} // namespace test
} // namespace gmx
