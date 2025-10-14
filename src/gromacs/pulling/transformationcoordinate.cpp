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

#include "gmxpre.h"

#include "gromacs/pulling/transformationcoordinate.h"

#include "gromacs/pulling/pull_internal.h"
#include "gromacs/pulling/pullcoordexpressionparser.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

namespace
{

//! Calculates the value a for transformation pull coordinate
double getTransformationPullCoordinateValue(pull_coord_work_t* coord)
{
    const int transformationPullCoordinateIndex = coord->params_.coordIndex;
    GMX_ASSERT(ssize(coord->transformationVariables) == transformationPullCoordinateIndex + 1,
               "We need as many variables as the transformation pull coordinate index plus one");
    double result = 0;
    try
    {
        result = coord->expressionParser.evaluate(coord->transformationVariables);
    }
#if HAVE_MUPARSER
    catch (mu::Parser::exception_type& e)
    {
        GMX_THROW(InconsistentInputError(
                formatString("failed to evaluate expression for transformation pull-coord%d: %s\n",
                             transformationPullCoordinateIndex + 1,
                             e.GetMsg().c_str())));
    }
#endif
    catch (std::exception& e)
    {
        GMX_THROW(InconsistentInputError(
                formatString("failed to evaluate expression for transformation pull-coord%d.\n"
                             "Last variable pull-coord-index: %d.\n"
                             "Message:  %s\n",
                             transformationPullCoordinateIndex + 1,
                             transformationPullCoordinateIndex + 1,
                             e.what())));
    }
    return result;
}

} // namespace

double getTransformationPullCoordinateValue(pull_coord_work_t*                coord,
                                            ArrayRef<const pull_coord_work_t> variableCoords,
                                            const double                      t)
{
    GMX_ASSERT(ssize(variableCoords) == coord->params_.coordIndex,
               "We need as many variables as the transformation pull coordinate index");
    int coordIndex = 0;
    for (const auto& variableCoord : variableCoords)
    {
        coord->transformationVariables[coordIndex++] = variableCoord.spatialData.value;
    }
    coord->transformationVariables[coordIndex] = t;

    return getTransformationPullCoordinateValue(coord);
}

/*! \brief Calculates and returns the derivative of a transformation pull coordinate from a dependent coordinate
 *
 * Note #1: this requires that getTransformationPullCoordinateValue() has been called
 * before with the current coordinates.
 *
 * Note #2: this method will not compute inner derivates. That is taken care of in the regular pull code
 *
 * \param[in] coord  The (transformation) coordinate to compute the value for
 * \param[in] variablePcrdIndex Pull coordinate index of a variable.
 */
static double computeDerivativeForTransformationPullCoord(pull_coord_work_t* coord, const int variablePcrdIndex)
{
    GMX_ASSERT(variablePcrdIndex >= 0 && variablePcrdIndex < coord->params_.coordIndex,
               "The variable index should be in range of the transformation coordinate");

    // epsilon for numerical differentiation.
    const double transformationPcrdValue = coord->spatialData.value;
    // Perform numerical differentiation of 1st order
    const double valueBackup = coord->transformationVariables[variablePcrdIndex];
    double       dx          = coord->params_.dx;
    coord->transformationVariables[variablePcrdIndex] += dx;
    double transformationPcrdValueEps = getTransformationPullCoordinateValue(coord);
    double derivative                 = (transformationPcrdValueEps - transformationPcrdValue) / dx;
    // reset pull coordinate value
    coord->transformationVariables[variablePcrdIndex] = valueBackup;
    return derivative;
}

void distributeTransformationPullCoordForce(pull_coord_work_t*               pcrd,
                                            gmx::ArrayRef<pull_coord_work_t> variableCoords)
{
    GMX_ASSERT(pcrd->params_.eGeom == PullGroupGeometry::Transformation,
               "We shouldn't end up here when not using a transformation pull coordinate.");
    GMX_ASSERT(ssize(variableCoords) == pcrd->params_.coordIndex,
               "We should have as many variable coords as the coord index of the transformation "
               "coordinate");

    const double transformationCoordForce = pcrd->scalarForce;

    for (auto& variableCoord : variableCoords)
    {
        const double derivative =
                computeDerivativeForTransformationPullCoord(pcrd, variableCoord.params_.coordIndex);
        const double variablePcrdForce = transformationCoordForce * derivative;
        /* Since we loop over all pull coordinates with smaller index, there can be ones
         * that are not referenced by the transformation coordinate. Avoid apply forces
         * on those by skipping application of zero force.
         */
        if (variablePcrdForce != 0)
        {
            if (debug)
            {
                fprintf(debug,
                        "Distributing force %4.4f for transformation coordinate %d to coordinate "
                        "%d with "
                        "force "
                        "%4.4f\n",
                        transformationCoordForce,
                        pcrd->params_.coordIndex,
                        variableCoord.params_.coordIndex,
                        variablePcrdForce);
            }
            // Note that we add to the force here, in case multiple biases act on the same pull
            // coord (although that is not recommended it should still work)
            variableCoord.scalarForce += variablePcrdForce;
        }
    }
}

} // namespace gmx
