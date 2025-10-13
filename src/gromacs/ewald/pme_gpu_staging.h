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

/*! \libinternal \file
 * \brief Defines the host-side PME GPU data structures.
 * \todo Some renaming/refactoring, which does not impair the performance:
 * -- bringing the function names up to guidelines
 * -- PmeGpuSettings -> PmeGpuTasks
 * -- refining GPU notation application (#2053)
 * -- renaming coefficients to charges (?)
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_ewald
 */

#ifndef GMX_EWALD_PME_GPU_STAGING_H
#define GMX_EWALD_PME_GPU_STAGING_H

#include <vector>

#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/math/vectypes.h"

#ifndef NUM_STATES
//! Number of FEP states.
#    define NUM_STATES 2
#endif

/*! \internal \brief
 * The PME GPU intermediate buffers structure, included in the main PME GPU structure by value.
 * Buffers are managed by the PME GPU module.
 */
struct PmeGpuStaging
{
    //! Host-side force buffer
    gmx::PaddedHostVector<gmx::RVec> h_forces;

    /*! \brief Virial and energy intermediate host-side buffer. Size is PME_GPU_VIRIAL_AND_ENERGY_COUNT. */
    float* h_virialAndEnergy[NUM_STATES];
    /*! \brief B-spline values intermediate host-side buffer. */
    float* h_splineModuli[NUM_STATES];

    /*! \brief Pointer to the host memory with B-spline values. Only used for host-side gather, or unit tests */
    float* h_theta;
    /*! \brief Pointer to the host memory with B-spline derivative values. Only used for host-side gather, or unit tests */
    float* h_dtheta;
    /*! \brief Pointer to the host memory with ivec atom gridline indices. Only used for host-side gather, or unit tests */
    int* h_gridlineIndices;
};

#endif
