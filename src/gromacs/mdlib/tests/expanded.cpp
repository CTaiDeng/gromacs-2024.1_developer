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

/*! \internal \file
 * \brief Tests for expanded ensemble
 *
 * This file contains unit tests for functions used by the expanded
 * ensemble.
 *
 * \todo Add more tests as the expanded ensemble implementation
 *       gets more modular (#3848).
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \author Michael Shirts <michael.shirts@colorado.edu>
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include <cmath>

#include <gtest/gtest.h>

#include "gromacs/mdlib/expanded_internal.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/enumerationhelpers.h"

#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

//! Test fixture accepting a value to pass into calculateAcceptanceWeight
class CalculateAcceptanceWeightSimple : public ::testing::Test, public ::testing::WithParamInterface<real>
{
};
// Check that unimplemented calculation modes throw
TEST_P(CalculateAcceptanceWeightSimple, UnknownCalculationModeThrows)
{
    for (auto calculationMode : gmx::EnumerationArray<LambdaWeightCalculation, bool>::keys())
    {
        if (calculationMode != LambdaWeightCalculation::Barker
            && calculationMode != LambdaWeightCalculation::Minvar
            && calculationMode != LambdaWeightCalculation::Metropolis)
        {
            EXPECT_THROW_GMX(calculateAcceptanceWeight(calculationMode, GetParam()), NotImplementedError);
        }
    }
}
// Check that implemented calculation modes don't throw
TEST_P(CalculateAcceptanceWeightSimple, KnownCalculationModeDoesNotThrow)
{
    EXPECT_NO_THROW(calculateAcceptanceWeight(LambdaWeightCalculation::Metropolis, GetParam()));
    EXPECT_NO_THROW(calculateAcceptanceWeight(LambdaWeightCalculation::Barker, GetParam()));
    EXPECT_NO_THROW(calculateAcceptanceWeight(LambdaWeightCalculation::Minvar, GetParam()));
}
// Barker and MinVar are expected to be equal
TEST_P(CalculateAcceptanceWeightSimple, BarkerAndMinVarAreIdentical)
{
    EXPECT_EQ(calculateAcceptanceWeight(LambdaWeightCalculation::Barker, GetParam()),
              calculateAcceptanceWeight(LambdaWeightCalculation::Minvar, GetParam()));
}

/*! \brief Test fixture accepting a calculation mode and an input value for
 *         calculateAcceptanceWeight as well as the expected output value
 */
using RegressionTuple = std::tuple<LambdaWeightCalculation, real, real>;
class CalculateAcceptanceWeightRangeRegression :
    public ::testing::Test,
    public ::testing::WithParamInterface<RegressionTuple>
{
};
// Check that output is as expected
TEST_P(CalculateAcceptanceWeightRangeRegression, ValuesMatch)
{
    const auto calculationMode = std::get<0>(GetParam());
    const auto inputValue      = std::get<1>(GetParam());
    const auto expectedOutput  = std::get<2>(GetParam());

    EXPECT_REAL_EQ(expectedOutput, calculateAcceptanceWeight(calculationMode, inputValue));
}

INSTANTIATE_TEST_SUITE_P(
        SimpleTests,
        CalculateAcceptanceWeightSimple,
        ::testing::Values(1., -1., 0., GMX_REAL_NEGZERO, GMX_REAL_EPS, -GMX_REAL_EPS, GMX_REAL_MAX, -GMX_REAL_MAX));
INSTANTIATE_TEST_SUITE_P(
        RegressionTests,
        CalculateAcceptanceWeightRangeRegression,
        ::testing::Values(
                RegressionTuple{ LambdaWeightCalculation::Metropolis, 0.0, 1.0 },
                RegressionTuple{ LambdaWeightCalculation::Metropolis, GMX_REAL_NEGZERO, 1.0 },
                RegressionTuple{ LambdaWeightCalculation::Metropolis, GMX_REAL_EPS, 1.0 },
                RegressionTuple{ LambdaWeightCalculation::Metropolis, -1.0, 1.0 },
                RegressionTuple{ LambdaWeightCalculation::Metropolis, -GMX_REAL_MAX, 1.0 },
                RegressionTuple{ LambdaWeightCalculation::Metropolis, 1.0, std::exp(-1.0) },
                RegressionTuple{ LambdaWeightCalculation::Metropolis, GMX_REAL_MAX, 0.0 },
                RegressionTuple{ LambdaWeightCalculation::Barker, 0.0, 0.5 },
                RegressionTuple{ LambdaWeightCalculation::Barker, GMX_REAL_NEGZERO, 0.5 },
                RegressionTuple{ LambdaWeightCalculation::Barker, GMX_REAL_EPS, 0.5 },
                RegressionTuple{ LambdaWeightCalculation::Barker, -1.0, 1.0 / (1.0 + std::exp(-1.0)) },
                RegressionTuple{ LambdaWeightCalculation::Barker, -GMX_REAL_MAX, 1.0 },
                RegressionTuple{ LambdaWeightCalculation::Barker, 1.0, 1.0 / (1.0 + std::exp(1.0)) },
                RegressionTuple{ LambdaWeightCalculation::Barker, GMX_REAL_MAX, 0.0 }));

} // namespace
} // namespace test
} // namespace gmx
