/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * \brief This file defines the PME CUDA-specific kernel parameter data structure.
 * \todo Rename the file (pme-gpu-types.cuh?), reconsider inheritance approach.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#ifndef GMX_EWALD_PME_CUH
#define GMX_EWALD_PME_CUH

#include "gromacs/math/vectypes.h" // for DIM

#include "pme_gpu_constants.h"
#include "pme_gpu_internal.h" // for GridOrdering
#include "pme_gpu_types.h"

/*! \brief \internal
 * An alias for PME parameters in CUDA.
 * \todo Remove if we decide to unify CUDA and OpenCL
 */
struct PmeGpuCudaKernelParams : PmeGpuKernelParamsBase
{
    // Place CUDA-specific stuff here
};

#endif
