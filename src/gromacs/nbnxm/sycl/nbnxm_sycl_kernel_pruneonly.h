/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 * Declares nbnxn sycl helper functions
 *
 * \ingroup module_nbnxm
 */
#ifndef GMX_NBNXM_SYCL_NBNXM_SYCL_KERNEL_PRUNEONLY_H
#define GMX_NBNXM_SYCL_NBNXM_SYCL_KERNEL_PRUNEONLY_H

// Forward declarations
namespace gmx
{
enum class InteractionLocality;
}
struct NbnxmGpu;

namespace Nbnxm
{
using gmx::InteractionLocality;

/*! \brief Launch SYCL NBNXM prune-only kernel.
 *
 * \param nb Non-bonded parameters.
 * \param iloc Interaction locality.
 * \param numParts Total number of rolling-prune parts.
 * \param part Number of the part to prune.
 * \param numSciInPart Number of superclusters in \p part.
 */
void launchNbnxmKernelPruneOnly(NbnxmGpu*                 nb,
                                const InteractionLocality iloc,
                                const int                 numParts,
                                const int                 part,
                                const int                 numSciInPart);

} // namespace Nbnxm

#endif // GMX_NBNXM_SYCL_NBNXM_SYCL_KERNEL_PRUNEONLY_H
