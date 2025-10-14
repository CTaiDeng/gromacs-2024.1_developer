/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 *  \brief
 *  Data management and kernel launch functions for nbnxm sycl.
 *
 *  \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "gromacs/nbnxm/gpu_common.h"
#include "gromacs/utility/exceptions.h"

#include "nbnxm_sycl_kernel.h"
#include "nbnxm_sycl_kernel_pruneonly.h"
#include "nbnxm_sycl_types.h"

namespace Nbnxm
{

void gpu_launch_kernel_pruneonly(NbnxmGpu* nb, const InteractionLocality iloc, const int numParts)
{
    gpu_plist* plist = nb->plist[iloc];

    if (plist->haveFreshList)
    {
        GMX_ASSERT(numParts == 1, "With first pruning we expect 1 part");

        /* Set rollingPruningNumParts to signal that it is not set */
        plist->rollingPruningNumParts = 0;
        plist->rollingPruningPart     = 0;
    }
    else
    {
        if (plist->rollingPruningNumParts == 0)
        {
            plist->rollingPruningNumParts = numParts;
        }
        else
        {
            GMX_ASSERT(numParts == plist->rollingPruningNumParts,
                       "It is not allowed to change numParts in between list generation steps");
        }
    }

    /* Use a local variable for part and update in plist, so we can return here
     * without duplicating the part increment code.
     */
    const int part = plist->rollingPruningPart;

    plist->rollingPruningPart++;
    if (plist->rollingPruningPart >= plist->rollingPruningNumParts)
    {
        plist->rollingPruningPart = 0;
    }

    /* Compute the number of list entries to prune in this pass */
    const int numSciInPart = (plist->nsci - part) / numParts;

    /* Don't launch the kernel if there is no work to do */
    if (numSciInPart <= 0)
    {
        plist->haveFreshList = false;
        return;
    }

    launchNbnxmKernelPruneOnly(nb, iloc, numParts, part, numSciInPart);

    if (plist->haveFreshList)
    {
        plist->haveFreshList = false;
        nb->didPrune[iloc]   = true; // Mark that pruning has been done
    }
    else
    {
        nb->didRollingPrune[iloc] = true; // Mark that rolling pruning has been done
    }
}

void gpu_launch_kernel(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const Nbnxm::InteractionLocality iloc)
{
    const NBParamGpu* nbp   = nb->nbparam;
    gpu_plist*        plist = nb->plist[iloc];

    if (canSkipNonbondedWork(*nb, iloc))
    {
        plist->haveFreshList = false;
        return;
    }

    if (nbp->useDynamicPruning && plist->haveFreshList)
    {
        // Prunes for rlistOuter and rlistInner, sets plist->haveFreshList=false
        gpu_launch_kernel_pruneonly(nb, iloc, 1);
    }

    if (plist->nsci == 0)
    {
        /* Don't launch an empty local kernel */
        return;
    }

    launchNbnxmKernel(nb, stepWork, iloc);
}

} // namespace Nbnxm
