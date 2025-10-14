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
 * \brief
 * Declares nbnxn sycl helper functions
 *
 * \ingroup module_nbnxm
 */
#ifndef GMX_NBNXM_SYCL_NBNXM_SYCL_KERNEL_H
#define GMX_NBNXM_SYCL_NBNXM_SYCL_KERNEL_H

// Forward declarations
namespace gmx
{
enum class InteractionLocality;
class StepWorkload;
} // namespace gmx
struct NbnxmGpu;

// Ensure any changes are in sync with device_management_sycl.cpp
#define SYCL_NBNXM_SUPPORTS_SUBGROUP_SIZE_8 (GMX_GPU_NB_CLUSTER_SIZE == 4)
#define SYCL_NBNXM_SUPPORTS_SUBGROUP_SIZE_32 (GMX_GPU_NB_CLUSTER_SIZE == 8)
#define SYCL_NBNXM_SUPPORTS_SUBGROUP_SIZE_64 \
    (GMX_GPU_NB_CLUSTER_SIZE == 8 && !(GMX_SYCL_HIPSYCL && !GMX_HIPSYCL_HAVE_HIP_TARGET))

namespace Nbnxm
{
using gmx::InteractionLocality;

/*! \brief Launch SYCL NBNXM kernel.
 *
 * \param nb Non-bonded parameters.
 * \param stepWork Workload flags for the current step.
 * \param iloc Interaction locality.
 */
void launchNbnxmKernel(NbnxmGpu* nb, const gmx::StepWorkload& stepWork, const InteractionLocality iloc);

} // namespace Nbnxm

#endif // GMX_NBNXM_SYCL_NBNXM_SYCL_KERNEL_H
