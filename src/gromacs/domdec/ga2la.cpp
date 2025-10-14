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

/*! \internal \file
 * \brief
 * Implements functions for mapping from global to local atom indices.
 *
 * \ingroup module_domdec
 *
 * \author Berk Hess <hess@kth.se>
 */

#include "gmxpre.h"

#include "ga2la.h"

/*! \brief Returns whether to use a direct list only
 *
 * There are two methods implemented for finding the local atom number
 * belonging to a global atom number:
 * 1) a simple, direct array
 * 2) a hash table consisting of list of linked lists indexed with
 *    the global number modulo mod.
 * Memory requirements:
 * 1) numAtomsTotal*2 ints
 * 2) numAtomsLocal*(2+1-2(1-e^-1/2))*4 ints
 * where numAtomsLocal is the number of atoms in the home + communicated zones.
 * Method 1 is faster for low parallelization, 2 for high parallelization.
 * We switch to method 2 when it uses less than half the memory method 1.
 */
static bool directListIsFaster(int numAtomsTotal, int numAtomsLocal)
{
    constexpr int c_numAtomsSmallRelativeToCache  = 1024;
    constexpr int c_memoryRatioHashedVersusDirect = 9;

    return (numAtomsTotal <= c_numAtomsSmallRelativeToCache
            || numAtomsTotal <= numAtomsLocal * c_memoryRatioHashedVersusDirect);
}

gmx_ga2la_t::gmx_ga2la_t(int numAtomsTotal, int numAtomsLocal) :
    usingDirect_(directListIsFaster(numAtomsTotal, numAtomsLocal))
{
    if (usingDirect_)
    {
        new (&(data_.direct)) std::vector<Entry>(numAtomsTotal, { -1, -1 });
    }
    else
    {
        new (&(data_.hashed)) gmx::HashedMap<Entry>(numAtomsLocal);
    }
}
