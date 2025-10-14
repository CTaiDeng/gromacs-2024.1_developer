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
 * This implements a fixture for calling calc_listed in gromacs
 * with nblib interaction data. The functionality implemented
 * in this file duplicates the corresponding NBLIB version and
 * exists for reference and testing purposes.
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */

#ifndef NBLIB_LISTEDFORCES_GMXCALCULATOR_H
#define NBLIB_LISTEDFORCES_GMXCALCULATOR_H

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/forcefieldparameters.h"
#include "gromacs/topology/idef.h"

#include "nblib/box.h"
#include "nblib/listed_forces/calculator.h"

namespace nblib
{

/*! \brief  an encapsulation class for gmx calc_listed
 *
 *  Holds all the necessary data to call calc_listed
 *  same ctor signature and behavior as the corresponding nblib
 *  ListedForceCalculator
 */
class ListedGmxCalculator
{
public:
    ListedGmxCalculator(const ListedInteractionData& interactions, int nP, int nThr, const Box& box);

    void compute(gmx::ArrayRef<const gmx::RVec>     x,
                 gmx::ArrayRef<gmx::RVec>           forces,
                 gmx::ArrayRef<gmx::RVec>           shiftForces,
                 ListedForceCalculator::EnergyType& energies,
                 bool                               usePbc);

    void compute(gmx::ArrayRef<const gmx::RVec>     x,
                 gmx::ArrayRef<gmx::RVec>           forces,
                 ListedForceCalculator::EnergyType& energies,
                 bool                               usePbc);

    [[nodiscard]] const InteractionDefinitions& getIdef() const;

private:
    int numParticles;
    int numThreads;

    Box box_;

    std::unique_ptr<InteractionDefinitions> idef;
    std::unique_ptr<gmx_ffparams_t>         ffparams;

    std::vector<gmx::RVec> shiftBuffer;
    std::vector<gmx::RVec> forceBuffer;

    gmx::ForceWithShiftForces shiftProxy;
    gmx::ForceWithVirial      virialProxy;
    gmx::ForceOutputs         forceOutputs; // yet another proxy

    t_forcerec   fr;
    t_disresdata disres_;
    t_fcdata     fcdata_;
    t_mdatoms    mdatoms_;

    t_pbc                          pbc;
    std::unique_ptr<gmx_wallcycle> wcycle;
    gmx_enerdata_t                 enerd;
    gmx::StepWorkload              stepWork;

    t_nrnb            nrnb;
    t_commrec         cr;
    std::vector<real> lambdaBuffer;

    std::unique_ptr<ListedForces> gmxListedForces_;
};

} // namespace nblib

#endif // NBLIB_LISTEDFORCES_GMXCALCULATOR_H
