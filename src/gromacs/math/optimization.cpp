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
 *
 * \brief Implements routines in optimization.h .
 *
 * \author Christian Blau <blau@kth.se>
 */

#include "gmxpre.h"

#include "gromacs/math/optimization.h"

#include "gromacs/math/neldermead.h"

namespace gmx
{

OptimisationResult nelderMead(const std::function<real(ArrayRef<const real>)>& functionToMinimize,
                              ArrayRef<const real>                             initalGuess,
                              real minimumRelativeSimplexLength,
                              int  maxSteps)
{
    // Set up the initial simplex, sorting vertices according to function value
    NelderMeadSimplex nelderMeadSimplex(functionToMinimize, initalGuess);

    // Run until maximum step size reached or algorithm is converged, e.g.,
    // the oriented simplex length is smaller or equal a given number.
    const real minimumSimplexLength = minimumRelativeSimplexLength * nelderMeadSimplex.orientedLength();
    for (int currentStep = 0;
         nelderMeadSimplex.orientedLength() > minimumSimplexLength && currentStep < maxSteps;
         ++currentStep)
    {

        // see if simplex can by improved by reflecing the worst vertex at the centroid
        const RealFunctionvalueAtCoordinate& reflectionPoint =
                nelderMeadSimplex.evaluateReflectionPoint(functionToMinimize);

        // Reflection point is not better than best simplex vertex so far
        // but better than second worst
        if ((nelderMeadSimplex.bestVertex().value_ <= reflectionPoint.value_)
            && (reflectionPoint.value_ < nelderMeadSimplex.secondWorstValue()))
        {
            nelderMeadSimplex.swapOutWorst(reflectionPoint);
            continue;
        }

        // If the reflection point is better than the best one see if simplex
        // can be further improved by continuing going in that direction
        if (reflectionPoint.value_ < nelderMeadSimplex.bestVertex().value_)
        {
            RealFunctionvalueAtCoordinate expansionPoint =
                    nelderMeadSimplex.evaluateExpansionPoint(functionToMinimize);
            if (expansionPoint.value_ < reflectionPoint.value_)
            {
                nelderMeadSimplex.swapOutWorst(expansionPoint);
            }
            else
            {
                nelderMeadSimplex.swapOutWorst(reflectionPoint);
            }
            continue;
        }

        // The reflection point was a poor choice, try contracting the
        // worst point coordinates using the centroid instead
        RealFunctionvalueAtCoordinate contractionPoint =
                nelderMeadSimplex.evaluateContractionPoint(functionToMinimize);
        if (contractionPoint.value_ < nelderMeadSimplex.worstVertex().value_)
        {
            nelderMeadSimplex.swapOutWorst(contractionPoint);
            continue;
        }

        // If neither expansion nor contraction of the worst point give a
        // good result shrink the whole simplex
        nelderMeadSimplex.shrinkSimplexPointsExceptBest(functionToMinimize);
    }

    return { nelderMeadSimplex.bestVertex().coordinate_, nelderMeadSimplex.bestVertex().value_ };
}

} // namespace gmx
