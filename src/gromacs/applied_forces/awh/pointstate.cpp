/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Implements the one method of the PointState class called only for one point per step.
 *
 * \author Viveca Lindahl
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_awh
 */

#include "gmxpre.h"

#include "pointstate.h"

namespace gmx
{

namespace
{

/*! \brief Returns the exponent c where exp(c) = exp(a) + exp(b).
 *
 * \param[in] a  First exponent.
 * \param[in] b  Second exponent.
 * \returns c.
 */
double expSum(double a, double b)
{
    return (a > b ? a : b) + std::log1p(std::exp(-std::fabs(a - b)));
}

} // namespace

void PointState::samplePmf(double convolvedBias)
{
    if (inTargetRegion())
    {
        logPmfSum_ = expSum(logPmfSum_, -convolvedBias);
        numVisitsIteration_ += 1;
    }
}

void PointState::updatePmfUnvisited(double bias)
{
    if (inTargetRegion())
    {
        logPmfSum_ = expSum(logPmfSum_, -bias);
    }
}

} // namespace gmx
