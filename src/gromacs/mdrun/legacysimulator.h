/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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

/*! \internal
 * \brief Declares the simulator interface for mdrun
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun
 */
#ifndef GMX_MDRUN_LEGACYSIMULATOR_H
#define GMX_MDRUN_LEGACYSIMULATOR_H

#include <cstdio>

#include <memory>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

#include "isimulator.h"

namespace gmx
{

//! Function type for simulator code.
using SimulatorFunctionType = void();

/*! \internal
 * \brief Struct to handle setting up and running the different simulation types.
 *
 * This struct is a mere aggregate of parameters to pass to run a
 * simulation, so that future changes to names and types of them consume
 * less time when refactoring other code.
 *
 * Having multiple simulation types as member functions isn't a good
 * design, and we definitely only intend one to be called, but the
 * goal is to make it easy to change the names and types of members
 * without having to make identical changes in several places in the
 * code. Once many of them have become modules, we should change this
 * approach.
 */
class LegacySimulator : public ISimulator, private LegacySimulatorData
{
private:
    //! Implements the normal MD simulations.
    SimulatorFunctionType do_md;
    //! Implements the rerun functionality.
    SimulatorFunctionType do_rerun;
    //! Implements steepest descent EM.
    SimulatorFunctionType do_steep;
    //! Implements conjugate gradient energy minimization
    SimulatorFunctionType do_cg;
    //! Implements onjugate gradient energy minimization using the L-BFGS algorithm
    SimulatorFunctionType do_lbfgs;
    //! Implements normal mode analysis
    SimulatorFunctionType do_nm;
    //! Implements test particle insertion
    SimulatorFunctionType do_tpi;
    //! Implements MiMiC QM/MM workflow
    SimulatorFunctionType do_mimic;
    // Use the constructor of the base class
    using LegacySimulatorData::LegacySimulatorData;

public:
    // Only builder can construct
    friend class SimulatorBuilder;

    /*! \brief Function to run the correct SimulatorFunctionType,
     * based on the .mdp integrator field. */
    void run() override;
};

} // namespace gmx

#endif // GMX_MDRUN_LEGACYSIMULATOR_H
