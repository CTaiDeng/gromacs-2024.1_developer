/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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

#include "gmxpre.h"

#include "kernels_reference/kernel_ref_prune.h"

#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/gmxassert.h"

#include "clusterdistancekerneltype.h"
#include "nbnxm_gpu.h"
#include "nbnxm_simd.h"
#include "pairlistset.h"
#include "pairlistsets.h"
#include "simd_prune_kernel.h"

void PairlistSets::dispatchPruneKernel(const gmx::InteractionLocality iLocality,
                                       const nbnxn_atomdata_t*        nbat,
                                       gmx::ArrayRef<const gmx::RVec> shift_vec)
{
    pairlistSet(iLocality).dispatchPruneKernel(nbat, shift_vec);
}

void PairlistSet::dispatchPruneKernel(const nbnxn_atomdata_t* nbat, gmx::ArrayRef<const gmx::RVec> shift_vec)
{
    const real rlistInner = params_.rlistInner;

    GMX_ASSERT(cpuLists_[0].ciOuter.size() >= cpuLists_[0].ci.size(),
               "Here we should either have an empty ci list or ciOuter should be >= ci");

    int gmx_unused nthreads = gmx_omp_nthreads_get(ModuleMultiThread::Nonbonded);
    GMX_ASSERT(nthreads == static_cast<gmx::Index>(cpuLists_.size()),
               "The number of threads should match the number of lists");
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < nthreads; i++)
    {
        NbnxnPairlistCpu* nbl = &cpuLists_[i];

        switch (getClusterDistanceKernelType(params_.pairlistType, *nbat))
        {
#if GMX_SIMD && GMX_HAVE_NBNXM_SIMD_4XM
            case ClusterDistanceKernelType::CpuSimd_4xM:
                nbnxmSimdPruneKernel<KernelLayout::r4xM>(nbl, *nbat, shift_vec, rlistInner);
                break;
#endif
#if GMX_SIMD && GMX_HAVE_NBNXM_SIMD_2XMM
            case ClusterDistanceKernelType::CpuSimd_2xMM:
                nbnxmSimdPruneKernel<KernelLayout::r2xMM>(nbl, *nbat, shift_vec, rlistInner);
                break;
#endif
            case ClusterDistanceKernelType::CpuPlainC:
                nbnxn_kernel_prune_ref(nbl, nbat, shift_vec, rlistInner);
                break;
            default: GMX_RELEASE_ASSERT(false, "kernel type not handled (yet)");
        }
    }
}

void nonbonded_verlet_t::dispatchPruneKernelCpu(const gmx::InteractionLocality iLocality,
                                                gmx::ArrayRef<const gmx::RVec> shift_vec) const
{
    pairlistSets_->dispatchPruneKernel(iLocality, nbat_.get(), shift_vec);
}

void nonbonded_verlet_t::dispatchPruneKernelGpu(int64_t step)
{
    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start_nocount(wcycle_, WallCycleSubCounter::LaunchGpuNonBonded);

    const bool stepIsEven =
            (pairlistSets().numStepsWithPairlist(step) % (2 * pairlistSets().params().mtsFactor) == 0);

    Nbnxm::gpu_launch_kernel_pruneonly(
            gpuNbv_,
            stepIsEven ? gmx::InteractionLocality::Local : gmx::InteractionLocality::NonLocal,
            pairlistSets().params().numRollingPruningParts);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuNonBonded);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpuPp);
}
