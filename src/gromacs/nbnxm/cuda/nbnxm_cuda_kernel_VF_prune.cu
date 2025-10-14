/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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

#include "gmxpre.h"

#include "gromacs/gpu_utils/cudautils.cuh"

#include "nbnxm_cuda_kernel_utils.cuh"
#include "nbnxm_cuda_types.h"

/* Top-level kernel generation: will generate through multiple
 * inclusion the following flavors for all kernel:
 * force and energy output without pair list pruning;
 */
#define PRUNE_NBL
#define CALC_ENERGIES
#define FUNCTION_DECLARATION_ONLY
#include "nbnxm_cuda_kernels.cuh"
#undef FUNCTION_DECLARATION_ONLY
#include "nbnxm_cuda_kernels.cuh"
#undef CALC_ENERGIES
#undef PRUNE_NBL
