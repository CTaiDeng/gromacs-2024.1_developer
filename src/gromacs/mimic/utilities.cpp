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

#include "gmxpre.h"

#include "utilities.h"

#include "gromacs/topology/topology.h"

std::vector<int> genQmmmIndices(const gmx_mtop_t& mtop)
{
    std::vector<int>     output;
    int                  global_at = 0;
    const unsigned char* grpnr =
            mtop.groups.groupNumbers[SimulationAtomGroupType::QuantumMechanics].data();
    for (const gmx_molblock_t& molb : mtop.molblock)
    {
        for (int mol = 0; mol < molb.nmol; ++mol)
        {
            for (int n_atom = 0; n_atom < mtop.moltype[molb.type].atoms.nr; ++n_atom)
            {
                if (!grpnr || !grpnr[global_at])
                {
                    output.push_back(global_at);
                }
                ++global_at;
            }
        }
    }
    return output;
}
