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
 *  \brief Declares data types for GPU timing
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \inlibraryapi
 */

#ifndef GMX_TIMING_GPU_TIMING_H
#define GMX_TIMING_GPU_TIMING_H

#include "gromacs/utility/enumerationhelpers.h"

/*! \internal \brief GPU kernel time and call count. */
struct gmx_kernel_timing_data_t
{
    double t; /**< Accumulated lapsed time */
    int    c; /**< Number of calls corresponding to the elapsed time */
};

/*! \internal \brief
 * PME GPU stages timing events indices, corresponding to the string in PMEStageNames in wallcycle.cpp.
 */
enum class PmeStage : int
{
    Spline = 0,
    Spread,
    SplineAndSpread,
    FftTransformR2C,
    Solve,
    FftTransformC2R,
    Gather,
    Count /* not a stage ID but a static array size */
};

/*! \internal \brief GPU timings for PME. */
struct gmx_wallclock_gpu_pme_t
{
    /* A separate PME structure to avoid refactoring the NB code for gmx_wallclock_gpu_t later
     * TODO: devise a better GPU timing data structuring.
     */
    /*! \brief Array of PME GPU timing data. */
    gmx::EnumerationArray<PmeStage, gmx_kernel_timing_data_t> timing;
};

/*! \internal \brief GPU NB timings for kernels and H2d/D2H transfers. */
struct gmx_wallclock_gpu_nbnxn_t
{
    gmx_kernel_timing_data_t ktime[2][2]; /**< table containing the timings of the four
                                                   versions of the nonbonded kernels: force-only,
                                                   force+energy, force+pruning, and force+energy+pruning */
    gmx_kernel_timing_data_t pruneTime; /**< table containing the timings of the 1st pass prune-only kernels */
    gmx_kernel_timing_data_t dynamicPruneTime; /**< table containing the timings of dynamic prune-only kernels */
    double                   nb_h2d_t; /**< host to device transfer time in nb calculation  */
    double                   nb_d2h_t; /**< device to host transfer time in nb calculation */
    int                      nb_c;     /**< total call count of the nonbonded gpu operations */
    double                   pl_h2d_t; /**< pair search step host to device transfer time */
    int                      pl_h2d_c; /**< pair search step  host to device transfer call count */
};

#endif
