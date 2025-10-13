/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/* \internal \file
 *
 * \brief Implements the atom distribution constructor
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */

#include "gmxpre.h"

#include "atomdistribution.h"

#include "gromacs/math/vec.h"

#include "domdec_internal.h"

/*! \brief Returns the total number of rank, determined from the DD grid dimensions */
static int numRanks(const ivec numCells)
{
    return numCells[XX] * numCells[YY] * numCells[ZZ];
}

AtomDistribution::AtomDistribution(const ivec numCells, int numAtomGroups, int numAtoms) :
    domainGroups(numRanks(numCells)),
    atomGroups(numAtomGroups),
    intBuffer(2 * numRanks(numCells)),
    rvecBuffer(numRanks(numCells) > c_maxNumRanksUseSendRecvForScatterAndGather ? numAtoms : 0)
{
    for (int d = 0; d < DIM; d++)
    {
        cellSizesBuffer[d].resize(numCells[d] + 1);
    }
}

void get_commbuffer_counts(AtomDistribution*         ma,
                           gmx::ArrayRef<const int>* counts,
                           gmx::ArrayRef<const int>* displacements)
{
    GMX_ASSERT(ma != nullptr, "Need a valid AtomDistribution struct (on the main rank)");

    /* Make the real (not rvec) count and displacement arrays */
    int  numRanks = ma->intBuffer.size() / 2;
    auto c        = gmx::makeArrayRef(ma->intBuffer).subArray(0, numRanks);
    auto d        = gmx::makeArrayRef(ma->intBuffer).subArray(numRanks, numRanks);
    for (int rank = 0; rank < numRanks; rank++)
    {
        c[rank] = ma->domainGroups[rank].numAtoms;
        d[rank] = (rank == 0 ? 0 : d[rank - 1] + c[rank - 1]);
    }

    *counts        = c;
    *displacements = d;
}
