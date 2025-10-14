/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * \brief Implements common util routines for different NBNXN GPU implementations
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_GPU_COMMON_UTILS_H
#define GMX_NBNXM_GPU_COMMON_UTILS_H

#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/range.h"

namespace Nbnxm
{

/*! \brief An early return condition for empty NB GPU workloads
 *
 * This is currently used for non-local kernels/transfers only.
 * Skipping the local kernel is more complicated, since the
 * local part of the force array also depends on the non-local kernel.
 * The skip of the local kernel is taken care of separately.
 */
static inline bool canSkipNonbondedWork(const NbnxmGpu& nb, InteractionLocality iloc)
{
    assert(nb.plist[iloc]);
    return (iloc == InteractionLocality::NonLocal && nb.plist[iloc]->nsci == 0);
}

/*! \brief Calculate atom range and return start index and length.
 *
 * \param[in] atomData Atom descriptor data structure
 * \param[in] atomLocality Atom locality specifier
 * \returns Range of indexes for selected locality.
 */
static inline gmx::Range<int> getGpuAtomRange(const NBAtomDataGpu* atomData, const AtomLocality atomLocality)
{
    assert(atomData);

    /* calculate the atom data index range based on locality */
    if (atomLocality == AtomLocality::Local)
    {
        return gmx::Range<int>(0, atomData->numAtomsLocal);
    }
    else if (atomLocality == AtomLocality::NonLocal)
    {
        return gmx::Range<int>(atomData->numAtomsLocal, atomData->numAtoms);
    }
    else
    {
        GMX_THROW(
                gmx::InconsistentInputError("Only Local and NonLocal atom localities can be used "
                                            "to get atom ranges in NBNXM."));
    }
}

} // namespace Nbnxm

#endif
