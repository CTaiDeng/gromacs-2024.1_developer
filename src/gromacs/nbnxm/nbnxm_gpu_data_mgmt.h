/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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

/*! \libinternal \file
 *  \brief Declare common functions for NBNXM GPU data management.
 *
 *  \author Artem Zhmurov <zhmurov@gmail.com>
 *
 *  \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_NBNXM_GPU_DATA_MGMT_H
#define GMX_NBNXM_NBNXM_GPU_DATA_MGMT_H

class DeviceContext;
struct interaction_const_t;
struct NBParamGpu;
struct PairlistParams;

namespace gmx
{
enum class InteractionLocality;
}

namespace Nbnxm
{

struct gpu_plist;

/*! \brief Initializes the NBNXM GPU data structures. */
void gpu_init_platform_specific(NbnxmGpu* nb);

/*! \brief Releases the NBNXM GPU data structures. */
void gpu_free_platform_specific(NbnxmGpu* nb);

#if GMX_GPU_CUDA
/*! Calculates working memory required for exclusive sum, used in neighbour list sorting */
void getExclusiveScanWorkingArraySize(size_t& scan_size, gpu_plist* d_plist, const DeviceStream& deviceStream);
#endif

} // namespace Nbnxm

#endif // GMX_NBNXM_NBNXM_GPU_DATA_MGMT_H
