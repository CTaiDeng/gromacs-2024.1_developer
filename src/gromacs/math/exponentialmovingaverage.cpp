/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * \brief
 * Implements routines to calculate an exponential moving average.
 *
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_math
 */
#include "gmxpre.h"

#include "gromacs/math/exponentialmovingaverage.h"

#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/keyvaluetree.h"

namespace gmx
{

//! Convert the exponential moving average state as key-value-tree object
void exponentialMovingAverageStateAsKeyValueTree(KeyValueTreeObjectBuilder            builder,
                                                 const ExponentialMovingAverageState& state)
{
    builder.addValue<real>("weighted-sum", state.weightedSum_);
    builder.addValue<real>("weighted-count", state.weightedCount_);
    builder.addValue<bool>("increasing", state.increasing_);
}

//! Sets the exponential moving average state from a key-value-tree object
ExponentialMovingAverageState exponentialMovingAverageStateFromKeyValueTree(const KeyValueTreeObject& object)
{
    const real weightedSum   = object["weighted-sum"].cast<real>();
    const real weightedCount = object["weighted-count"].cast<real>();
    const bool increasing    = object["increasing"].cast<bool>();
    return { weightedSum, weightedCount, increasing };
}

ExponentialMovingAverage::ExponentialMovingAverage(real timeConstant,
                                                   const ExponentialMovingAverageState& state) :
    state_(state)
{
    if (timeConstant < 1)
    {
        GMX_THROW(InconsistentInputError(
                "Lag time may not be negative or zero for exponential moving averages."));
    }
    inverseTimeConstant_ = 1. / timeConstant;
}

void ExponentialMovingAverage::updateWithDataPoint(real dataPoint)
{
    state_.weightedSum_   = dataPoint + (1 - inverseTimeConstant_) * state_.weightedSum_;
    state_.weightedCount_ = 1 + (1 - inverseTimeConstant_) * state_.weightedCount_;

    state_.increasing_ = dataPoint * state_.weightedCount_ > state_.weightedSum_;
}

const ExponentialMovingAverageState& ExponentialMovingAverage::state() const
{
    return state_;
}

real ExponentialMovingAverage::biasCorrectedAverage() const
{
    return state_.weightedSum_ / state_.weightedCount_;
}

bool ExponentialMovingAverage::increasing() const
{
    return state_.increasing_;
}

real ExponentialMovingAverage::inverseTimeConstant() const
{
    return inverseTimeConstant_;
}

} // namespace gmx
