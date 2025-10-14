/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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

/*! \inpublicapi \file
 * \brief
 * Implements nblib simulation box
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#ifndef NBLIB_GMXBACKENDDATA_H
#define NBLIB_GMXBACKENDDATA_H

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/listoflists.h"
#include "gromacs/utility/range.h"

#include "nblib/kerneloptions.h"
#include "nblib/nbnxmsetuphelpers.h"

namespace nblib
{

/*! \brief GROMACS non-bonded force calculation backend
 *
 * This class encapsulates the various GROMACS data structures and their interplay
 * from the NBLIB user. The class is a private member of the ForceCalculator and
 * is not intended for the public interface.
 *
 * Handles the task of storing the simulation problem description using the internal
 * representation used within GROMACS. It currently supports short range non-bonded
 * interactions (PP) on a single node CPU.
 *
 */
class GmxBackendData
{
public:
    GmxBackendData() = default;
    GmxBackendData(const NBKernelOptions& options,
                   int                    numEnergyGroups,
                   gmx::ArrayRef<int>     exclusionRanges,
                   gmx::ArrayRef<int>     exclusionElements) :
        numThreads_(options.numOpenMPThreads)
    {
        // Set hardware params from the execution context
        setGmxNonBondedNThreads(options.numOpenMPThreads);

        // Set interaction constants struct
        interactionConst_ = createInteractionConst(options);

        // Set up StepWorkload data
        stepWork_ = createStepWorkload();

        // Set up gmx_enerdata_t (holds energy information)
        enerd_ = gmx_enerdata_t{ numEnergyGroups, nullptr };

        // Construct pair lists
        std::vector<int> exclusionRanges_(exclusionRanges.begin(), exclusionRanges.end());
        std::vector<int> exclusionElements_(exclusionElements.begin(), exclusionElements.end());
        exclusions_ = gmx::ListOfLists<int>(std::move(exclusionRanges_), std::move(exclusionElements_));
    }

    //! exclusions in gmx format
    gmx::ListOfLists<int> exclusions_;

    //! Non-Bonded Verlet object for force calculation
    std::unique_ptr<nonbonded_verlet_t> nbv_;

    //! Only shift_vec is used
    t_forcerec forcerec_;

    //! Parameters for various interactions in the system
    interaction_const_t interactionConst_;

    //! Tasks to perform in an MD Step
    gmx::StepWorkload stepWork_;

    gmx::SimulationWorkload simulationWork_;

    //! Energies of different interaction types; currently only needed as an argument for dispatchNonbondedKernel
    gmx_enerdata_t enerd_{ 1, nullptr };

    //! Non-bonded flop counter; currently only needed as an argument for dispatchNonbondedKernel
    t_nrnb nrnb_;

    //! Number of OpenMP threads to use
    int numThreads_;

    //! Keep track of whether updatePairlist has been called at least once
    bool updatePairlistCalled{ false };
};

} // namespace nblib
#endif // NBLIB_GMXBACKENDDATA_H
