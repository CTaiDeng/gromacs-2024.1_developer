/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * Declares parameters needed to evaluate forces and energies for density fitting
 *
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_DENSITYFITTINGPARAMETERS_H
#define GMX_APPLIED_FORCES_DENSITYFITTINGPARAMETERS_H

#include <string>
#include <vector>

#include "gromacs/math/densityfit.h"
#include "gromacs/utility/basedefinitions.h"

#include "densityfittingamplitudelookup.h"

namespace gmx
{

/*! \internal
 * \brief Holding all directly user-provided parameters for density fitting.
 *
 * Also used for setting all default parameters.
 */
struct DensityFittingParameters
{
    //! Indicate if density fitting is active
    bool active_ = false;
    //! Indices of the atoms that shall be fit to the density
    std::vector<Index> indices_;
    //! Determines how to measure similarity between simulated and reference density
    DensitySimilarityMeasureMethod similarityMeasureMethod_ = DensitySimilarityMeasureMethod::innerProduct;
    //! Determines with what weight atoms are spread
    DensityFittingAmplitudeMethod amplitudeLookupMethod_ = DensityFittingAmplitudeMethod::Unity;
    //! The force constant to be used for the density fitting
    real forceConstant_ = 1e9;
    //! The spreading width used for the gauss transform of atoms onto the density grid
    real gaussianTransformSpreadingWidth_ = 0.2;
    //! The spreading range for spreading atoms onto the grid in multiples of the spreading width
    real gaussianTransformSpreadingRangeInMultiplesOfWidth_ = 4.0;
    //! Apply density fitting forces only every n-steps
    std::int64_t calculationIntervalInSteps_ = 1;
    //! Normalize reference and simulated densities
    bool normalizeDensities_ = true;
    //! Perform adaptive force scaling during the simulation
    bool adaptiveForceScaling_ = false;
    //! The time constant for the adaptive force scaling in ps
    real adaptiveForceScalingTimeConstant_ = 4;
    //! Translation of the structure, so that the coordinates that are fitted are x+translation
    std::string translationString_;
    //! Linear transformation of the structure, so that the coordinates that are fitted are Matrix * x
    std::string transformationMatrixString_;
};

/*!\brief Check if two structs holding density fitting parameters are equal.
 *
 * \param[in] lhs left hand side to be compared
 * \param[in] rhs right hand side to be compared
 * \returns true if all elements in DensityFittingParameters are equal, else false
 */
bool operator==(const DensityFittingParameters& lhs, const DensityFittingParameters& rhs);

/*!\brief Check if two structs holding density fitting parameters are not equal.
 *
 * \param[in] lhs left hand side to be compared
 * \param[in] rhs right hand side to be compared
 * \returns true if lhs is not equal rhs
 */
bool operator!=(const DensityFittingParameters& lhs, const DensityFittingParameters& rhs);

} // namespace gmx

#endif // GMX_APPLIED_FORCES_DENSITYFITTINGPARAMETERS_H
