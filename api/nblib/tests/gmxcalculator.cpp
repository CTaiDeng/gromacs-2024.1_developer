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
 * This implements basic nblib utility tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include <gtest/gtest.h>

#include "gromacs/utility/arrayref.h"

#include "nblib/gmxcalculatorcpu.h"
#include "nblib/kerneloptions.h"
#include "nblib/simulationstate.h"

#include "testhelpers.h"
#include "testsystems.h"

namespace nblib
{
namespace test
{
namespace
{
TEST(NBlibTest, GmxForceCalculatorCanCompute)
{
    ArgonSimulationStateBuilder argonSystemBuilder(fftypes::GROMOS43A1);
    SimulationState             simState = argonSystemBuilder.setupSimulationState();
    NBKernelOptions             options  = NBKernelOptions();
    options.nbnxmSimd                    = SimdKernels::SimdNo;

    std::unique_ptr<GmxNBForceCalculatorCpu> gmxForceCalculator =
            setupGmxForceCalculatorCpu(simState.topology(), options);
    gmxForceCalculator->updatePairlist(simState.coordinates(), simState.box());

    EXPECT_NO_THROW(gmxForceCalculator->compute(simState.coordinates(), simState.box(), simState.forces()));
}

TEST(NBlibTest, ArgonVirialsAreCorrect)
{
    ArgonSimulationStateBuilder argonSystemBuilder(fftypes::OPLSA);
    SimulationState             simState = argonSystemBuilder.setupSimulationState();
    NBKernelOptions             options  = NBKernelOptions();
    options.nbnxmSimd                    = SimdKernels::SimdNo;
    std::unique_ptr<GmxNBForceCalculatorCpu> gmxForceCalculator =
            setupGmxForceCalculatorCpu(simState.topology(), options);
    gmxForceCalculator->updatePairlist(simState.coordinates(), simState.box());

    std::vector<real> virialArray(9, 0.0);

    gmxForceCalculator->compute(simState.coordinates(), simState.box(), simState.forces(), virialArray);

    RefDataChecker virialsOutputTest(1e-7);
    virialsOutputTest.testArrays<real>(virialArray, "Virials");
}

TEST(NBlibTest, ArgonEnergiesAreCorrect)
{
    ArgonSimulationStateBuilder argonSystemBuilder(fftypes::OPLSA);
    SimulationState             simState = argonSystemBuilder.setupSimulationState();
    NBKernelOptions             options  = NBKernelOptions();
    options.nbnxmSimd                    = SimdKernels::SimdNo;
    std::unique_ptr<GmxNBForceCalculatorCpu> gmxForceCalculator =
            setupGmxForceCalculatorCpu(simState.topology(), options);
    gmxForceCalculator->updatePairlist(simState.coordinates(), simState.box());

    // number of energy kinds is 5: COULSR, LJSR, BHAMSR, COUL14, LJ14,
    std::vector<real> energies(5, 0.0);

    gmxForceCalculator->compute(
            simState.coordinates(), simState.box(), simState.forces(), gmx::ArrayRef<real>{}, energies);

    RefDataChecker energiesOutputTest(5e-5);
    energiesOutputTest.testArrays<real>(energies, "Argon energies");
}

TEST(NBlibTest, SpcMethanolEnergiesAreCorrect)
{
    SpcMethanolSimulationStateBuilder spcMethanolSystemBuilder;
    SimulationState                   simState = spcMethanolSystemBuilder.setupSimulationState();
    NBKernelOptions                   options  = NBKernelOptions();
    options.nbnxmSimd                          = SimdKernels::SimdNo;
    std::unique_ptr<GmxNBForceCalculatorCpu> gmxForceCalculator =
            setupGmxForceCalculatorCpu(simState.topology(), options);
    gmxForceCalculator->updatePairlist(simState.coordinates(), simState.box());

    // number of energy kinds is 5: COULSR, LJSR, BHAMSR, COUL14, LJ14,
    std::vector<real> energies(5, 0.0);

    gmxForceCalculator->compute(
            simState.coordinates(), simState.box(), simState.forces(), gmx::ArrayRef<real>{}, energies);

    RefDataChecker energiesOutputTest(5e-5);
    energiesOutputTest.testArrays<real>(energies, "SPC-methanol energies");
}

} // namespace
} // namespace test
} // namespace nblib
