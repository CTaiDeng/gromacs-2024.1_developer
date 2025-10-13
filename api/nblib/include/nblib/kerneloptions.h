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

/*! \inpublicapi \file
 * \brief
 * Implements nblib kernel setup options
 *
 * \author Berk Hess <hess@kth.se>
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#ifndef NBLIB_KERNELOPTIONS_H
#define NBLIB_KERNELOPTIONS_H

#include <memory>

#include "nblib/basicdefinitions.h"

namespace nblib
{

//! Enum for selecting the SIMD kernel type
enum class SimdKernels : int
{
    SimdAuto,
    SimdNo,
    Simd4XM,
    Simd2XMM,
    Count
};

//! Enum for selecting the combination rule
enum class CombinationRule : int
{
    Geometric,
    LorentzBerthelot,
    None,
    Count
};

//! Enum for selecting coulomb type
enum class CoulombType : int
{
    Pme,
    Cutoff,
    ReactionField,
    Count
};

/*! \internal \brief
 * The options for the nonbonded kernel caller
 */
struct NBKernelOptions final
{
    //! Whether to use a GPU, currently GPUs are not supported
    bool useGpu = false;
    //! The number of OpenMP threads to use
    int numOpenMPThreads = 1;
    //! The SIMD type for the kernel
    SimdKernels nbnxmSimd = SimdKernels::SimdAuto;
    //! The LJ combination rule
    CombinationRule ljCombinationRule = CombinationRule::Geometric;
    //! The pairlist and interaction cut-off
    real pairlistCutoff = 1.0;
    //! The Coulomb interaction function
    CoulombType coulombType = CoulombType::Pme;
    //! Whether to use tabulated PME grid correction instead of analytical, not applicable with simd=no
    bool useTabulatedEwaldCorr = false;
    //! The number of iterations for each kernel
    int numIterations = 100;
    //! The time step
    real timestep = 0.001;
};

} // namespace nblib

#endif // NBLIB_KERNELOPTIONS_H
