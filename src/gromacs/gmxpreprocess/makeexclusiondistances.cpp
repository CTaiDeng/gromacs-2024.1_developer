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

#include "gmxpre.h"

#include "makeexclusiondistances.h"

#include <string>
#include <vector>

#include "gromacs/topology/atomprop.h"
#include "gromacs/topology/atoms.h"

std::vector<real> makeExclusionDistances(const t_atoms* a, AtomProperties* aps, real defaultDistance, real scaleFactor)
{
    std::vector<real> exclusionDistances;

    if (a != nullptr)
    {
        exclusionDistances.reserve(a->nr);
        for (int i = 0; i < a->nr; ++i)
        {
            real value;
            if (!aps->setAtomProperty(epropVDW,
                                      std::string(*(a->resinfo[a->atom[i].resind].name)),
                                      std::string(*(a->atomname[i])),
                                      &value))
            {
                value = defaultDistance;
            }
            else
            {
                value *= scaleFactor;
            }
            exclusionDistances.push_back(value);
        }
    }
    return exclusionDistances;
}
