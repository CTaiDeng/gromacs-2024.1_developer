/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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

#ifndef GMX_AWH_TEST_SETUP_H
#define GMX_AWH_TEST_SETUP_H

#include "gmxpre.h"

#include <memory>
#include <vector>

#include "gromacs/applied_forces/awh/bias.h"
#include "gromacs/mdtypes/awh_params.h"

namespace gmx
{

template<typename>
class ArrayRef;

namespace test
{

/*! \internal \brief
 * Prepare a memory buffer with serialized AwhDimParams.
 */
std::vector<char> awhDimParamSerialized(
        AwhCoordinateProviderType inputCoordinateProvider = AwhCoordinateProviderType::Pull,
        int                       inputCoordIndex         = 0,
        double                    inputOrigin             = 0.5,
        double                    inputEnd                = 1.5,
        double                    inputPeriod             = 0,
        // Correction for removal of GaussianGeometryFactor/2 in histogram size
        double inputDiffusion = 0.1 / (0.144129616073222 * 2));

/*! \internal \brief
 * Struct that gathers all input for setting up and using a Bias
 */
struct AwhTestParameters
{
    explicit AwhTestParameters(ISerializer* serializer);
    //! Move constructor
    AwhTestParameters(AwhTestParameters&& o) noexcept :
        beta(o.beta), awhParams(std::move(o.awhParams)), dimParams(std::move(o.dimParams))
    {
    }
    //! 1/(kB*T).
    double beta;

    //! AWH parameters, this is the struct to actually use.
    AwhParams awhParams;
    //! Dimension parameters for setting up Bias.
    std::vector<DimParams> dimParams;
};

/*! \brief
 * Helper function to set up the C-style AWH parameters for the test.
 *
 * Builds the test input data from serialized data.
 */
AwhTestParameters getAwhTestParameters(AwhHistogramGrowthType            eawhgrowth,
                                       AwhPotentialType                  eawhpotential,
                                       ArrayRef<const std::vector<char>> dimensionParameterBuffers,
                                       bool                              inputUserData,
                                       double                            beta,
                                       bool                              useAwhFep,
                                       double                            inputErrorScaling,
                                       int                               numFepLambdaStates,
                                       int                               biasShareGroup = 0,
                                       AwhTargetType eTargetType         = AwhTargetType::Constant,
                                       bool          scaleTargetByMetric = false);

} // namespace test
} // namespace gmx

#endif
