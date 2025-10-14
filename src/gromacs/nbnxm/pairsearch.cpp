/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * Implements the PairSearch class
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#include "gmxpre.h"

#include "pairsearch.h"

#include "gromacs/mdtypes/nblist.h"

#include "pairlist.h"


void SearchCycleCounting::printCycles(FILE* fp, gmx::ArrayRef<const PairsearchWork> work) const
{
    fprintf(fp, "\n");
    fprintf(fp,
            "ns %4d grid %4.1f search %4.1f",
            cc_[enbsCCgrid].count(),
            cc_[enbsCCgrid].averageMCycles(),
            cc_[enbsCCsearch].averageMCycles());

    if (work.size() > 1)
    {
        if (cc_[enbsCCcombine].count() > 0)
        {
            fprintf(fp, " comb %5.2f", cc_[enbsCCcombine].averageMCycles());
        }
        fprintf(fp, " s. th");
        for (const PairsearchWork& workEntry : work)
        {
            fprintf(fp, " %4.1f", workEntry.cycleCounter.averageMCycles());
        }
    }
    fprintf(fp, "\n");
}

#ifndef DOXYGEN

PairsearchWork::PairsearchWork() :
    cp0({ { 0 } }), ndistc(0), nbl_fep(std::make_unique<t_nblist>()), cp1({ { 0 } })
{
}

#endif // !DOXYGEN

PairsearchWork::~PairsearchWork() = default;

PairSearch::PairSearch(const PbcType             pbcType,
                       const bool                doTestParticleInsertion,
                       const ivec*               numDDCells,
                       const gmx_domdec_zones_t* ddZones,
                       const PairlistType        pairlistType,
                       const bool                haveFep,
                       const int                 maxNumThreads,
                       gmx::PinningPolicy        pinningPolicy) :
    gridSet_(pbcType, doTestParticleInsertion, numDDCells, ddZones, pairlistType, haveFep, maxNumThreads, pinningPolicy),
    work_(maxNumThreads)
{
    cycleCounting_.recordCycles_ = (getenv("GMX_NBNXN_CYCLE") != nullptr);
}
