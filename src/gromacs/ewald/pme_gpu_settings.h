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

/*! \libinternal \file
 * \brief Defines the PME GPU settings data structures.
 * \todo Some renaming/refactoring, which does not impair the performance:
 * -- PmeGpuSettings -> PmeGpuTasks
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_ewald
 */

#ifndef GMX_EWALD_PME_GPU_SETTINGS_H
#define GMX_EWALD_PME_GPU_SETTINGS_H

#include "gromacs/gpu_utils/gpu_utils.h" // for GpuApiCallBehavior

enum class ThreadsPerAtom : int;

/*! \internal \brief
 * The PME GPU settings structure, included in the main PME GPU structure by value.
 */
struct PmeGpuSettings
{
    /* Permanent settings set on initialization */
    /*! \brief A boolean which tells if the solving is performed on GPU. Currently always true */
    bool performGPUSolve;
    /*! \brief A boolean which tells if the gathering is performed on GPU. Currently always true */
    bool performGPUGather;
    /*! \brief A boolean which tells if the FFT is performed on GPU. Currently true for a single MPI rank. */
    bool performGPUFFT;
    /*! \brief A convenience boolean which tells if PME decomposition is used. */
    bool useDecomposition;
    /*! \brief True if PME forces are reduced on-GPU, false if reduction is done on the CPU;
     *  in the former case transfer does not need to happen.
     *
     *  Note that this flag may change per-step.
     */
    bool useGpuForceReduction;

    /*! \brief A boolean which tells if any PME GPU stage should copy all of its outputs to the
     * host. Only intended to be used by the test framework.
     */
    bool copyAllOutputs;
    /*! \brief An enum which tells whether most PME GPU D2H/H2D data transfers should be synchronous. */
    GpuApiCallBehavior transferKind;
    /*! \brief
     *  Controls whether we use order (i.e. 4) threads per atom for the GPU
     *  or order*order (i.e. 16) threads per atom.
     *
     *  Currently ThreadsPerAtom::Order is only supported by CUDA.
     */
    ThreadsPerAtom threadsPerAtom;
    /*! \brief
     * Currently only supported by CUDA.
     * Controls if we should recalculate the splines in the gather or
     * save the values in the spread and reload in the gather.
     */
    bool recalculateSplines;
};

#endif
