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

/*! \internal \file
 *
 * \brief
 * Declares the free-energy kernel dispatch class
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */
#ifndef GMX_NBNXM_FREEENERGYDISPATCH_H
#define GMX_NBNXM_FREEENERGYDISPATCH_H

#include <memory>

#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/threaded_force_buffer.h"
#include "gromacs/utility/arrayref.h"

struct gmx_enerdata_t;
struct gmx_wallcycle;
struct interaction_const_t;
class PairlistSets;
struct t_lambda;
struct t_nrnb;

namespace gmx
{
template<typename>
class ArrayRefWithPadding;
class ForceWithShiftForces;
class StepWorkload;
} // namespace gmx

/*! \internal
 *  \brief Temporary data and methods for handling dispatching of the nbnxm free-energy kernels
 */
class FreeEnergyDispatch
{
public:
    //! Constructor
    FreeEnergyDispatch(int numEnergyGroups);

    //! Sets up the threaded force buffer and the reduction, should be called after constructing the pair lists
    void setupFepThreadedForceBuffer(int numAtomsForce, const PairlistSets& pairlistSets);

    //! Dispatches the non-bonded free-energy kernels, thread parallel and reduces the output
    void dispatchFreeEnergyKernels(const PairlistSets&                              pairlistSets,
                                   const gmx::ArrayRefWithPadding<const gmx::RVec>& coords,
                                   gmx::ForceWithShiftForces*     forceWithShiftForces,
                                   bool                           useSimd,
                                   int                            ntype,
                                   const interaction_const_t&     ic,
                                   gmx::ArrayRef<const gmx::RVec> shiftvec,
                                   gmx::ArrayRef<const real>      nbfp,
                                   gmx::ArrayRef<const real>      nbfp_grid,
                                   gmx::ArrayRef<const real>      chargeA,
                                   gmx::ArrayRef<const real>      chargeB,
                                   gmx::ArrayRef<const int>       typeA,
                                   gmx::ArrayRef<const int>       typeB,
                                   gmx::ArrayRef<const real>      lambda,
                                   gmx_enerdata_t*                enerd,
                                   const gmx::StepWorkload&       stepWork,
                                   t_nrnb*                        nrnb,
                                   gmx_wallcycle*                 wcycle);

private:
    //! Temporary array for storing foreign lambda group pair energies
    gmx_grppairener_t foreignGroupPairEnergies_;

    //! Threaded force buffer for nonbonded FEP
    gmx::ThreadedForceBuffer<gmx::RVec> threadedForceBuffer_;
    //! Threaded buffer for nonbonded FEP foreign energies and dVdl, no forces, so numAtoms = 0
    gmx::ThreadedForceBuffer<gmx::RVec> threadedForeignEnergyBuffer_;
};

#endif // GMX_NBNXM_FREEENERGYDISPATCH_H
