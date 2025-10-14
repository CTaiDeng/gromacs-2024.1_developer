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
 * Implements the PairlistParams constructor
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#include "gmxpre.h"

#include "pairlistparams.h"

#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/utility/gmxassert.h"

#include "nbnxm_geometry.h"


PairlistParams::PairlistParams(const Nbnxm::KernelType kernelType,
                               const bool              haveFep,
                               const real              rlist,
                               const bool              haveMultipleDomains) :
    haveFep_(haveFep),
    rlistOuter(rlist),
    rlistInner(rlist),
    haveMultipleDomains_(haveMultipleDomains),
    useDynamicPruning(false),
    mtsFactor(1),
    nstlistPrune(-1),
    numRollingPruningParts(1),
    lifetime(-1)
{
    if (!Nbnxm::kernelTypeUsesSimplePairlist(kernelType))
    {
        pairlistType = PairlistType::HierarchicalNxN;
    }
    else
    {
        switch (Nbnxm::JClusterSizePerKernelType[kernelType])
        {
            case 2: pairlistType = PairlistType::Simple4x2; break;
            case 4: pairlistType = PairlistType::Simple4x4; break;
            case 8: pairlistType = PairlistType::Simple4x8; break;
            default: GMX_RELEASE_ASSERT(false, "Kernel type does not have a pairlist type");
        }
    }
}
