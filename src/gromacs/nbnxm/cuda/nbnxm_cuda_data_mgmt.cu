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

/*! \file
 *  \brief Define CUDA implementation of nbnxn_gpu_data_mgmt.h
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 */
#include "gmxpre.h"

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

// TODO We would like to move this down, but the way NbnxmGpu
//      is currently declared means this has to be before gpu_types.h
#include "nbnxm_cuda_types.h"

// TODO Remove this comment when the above order issue is resolved
#include <cub/device/device_scan.cuh>

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/pmalloc.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/gridset.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/nbnxm_gpu_data_mgmt.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"

#include "nbnxm_cuda.h"
/* Required to stop gcc emitting multiple definition warnings as cuda_fp16.h, which is included by
 * device_scan.cuh, doesn't undef __WSB_DEPRECATION_MESSAGE and this is later redefined in
 * device_atomic_functions.h used by nbnxm_cuda_types.h. Seen in cuda 10 and 11 with gcc-11. */
#undef __WSB_DEPRECATION_MESSAGE

namespace Nbnxm
{

/* This is a heuristically determined parameter for the Kepler
 * and Maxwell architectures for the minimum size of ci lists by multiplying
 * this constant with the # of multiprocessors on the current device.
 * Since the maximum number of blocks per multiprocessor is 16, the ideal
 * count for small systems is 32 or 48 blocks per multiprocessor. Because
 * there is a bit of fluctuations in the generated block counts, we use
 * a target of 44 instead of the ideal value of 48.
 */

#if GMX_PTX_ARCH <= 700
static const unsigned int gpu_min_ci_balanced_factor = 44;
#else
/* Updated benchmarking on Ampere, Ada, Hopper shows the ideal count is
 * between 61 and 83 depending on chip */
static const unsigned int gpu_min_ci_balanced_factor = 61;
#endif


void gpu_init_platform_specific(NbnxmGpu* /* nb */)
{
    /* set the kernel type for the current GPU */
    /* pick L1 cache configuration */
    cuda_set_cacheconfig();
}

void gpu_free_platform_specific(NbnxmGpu* /* nb */)
{
    // Nothing specific in CUDA
}

int gpu_min_ci_balanced(NbnxmGpu* nb)
{
    return nb != nullptr ? gpu_min_ci_balanced_factor * nb->deviceContext_->deviceInfo().prop.multiProcessorCount
                         : 0;
}

/* Calculate size of working memory required for exclusive sum, part of sorting the neighbour list,
 * by calling exclusive sum with nullptr */
void getExclusiveScanWorkingArraySize(size_t& scan_size, gpu_plist* d_plist, const DeviceStream& deviceStream)
{
    cub::DeviceScan::ExclusiveSum(nullptr,
                                  scan_size,
                                  d_plist->sorting.sciHistogram,
                                  d_plist->sorting.sciOffset,
                                  c_sciHistogramSize,
                                  deviceStream.stream());
}

} // namespace Nbnxm
