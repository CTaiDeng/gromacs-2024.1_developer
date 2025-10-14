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
 *  Wrapper for the backend-specific coordinate layout conversion functionality
 *
 *  \ingroup module_nbnxm
 */
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gputraits.h"

class DeviceStream;
struct NbnxmGpu;

namespace Nbnxm
{

class Grid;

/*! \brief Launch coordinate layout conversion kernel
 *
 * \param[in]     grid          Pair-search grid.
 * \param[in,out] nb            Nbnxm main structure.
 * \param[in]     d_x           Source atom coordinates.
 * \param[in]     deviceStream  Device stream for kernel submission.
 * \param[in]     numColumnsMax Max. number of columns per grid for offset calculation in \p nb.
 * \param[in]     gridId        Grid index for offset calculation in \p nb.
 */
void launchNbnxmKernelTransformXToXq(const Grid&          grid,
                                     NbnxmGpu*            nb,
                                     DeviceBuffer<Float3> d_x,
                                     const DeviceStream&  deviceStream,
                                     unsigned int         numColumnsMax,
                                     int                  gridId);

} // namespace Nbnxm
