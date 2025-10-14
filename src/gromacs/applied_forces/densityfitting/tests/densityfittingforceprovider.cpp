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
 * Tests amplitude lookup for density fitting
 *
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gromacs/applied_forces/densityfitting/densityfittingforceprovider.h"

#include <gtest/gtest.h>

namespace gmx
{

TEST(DensityFittingForceProviderState, RoundTripSaving)
{
    DensityFittingForceProviderState state;
    // set-up state
    state.adaptiveForceConstantScale_                   = 1.0;
    state.stepsSinceLastCalculation_                    = 0;
    state.exponentialMovingAverageState_.increasing_    = false;
    state.exponentialMovingAverageState_.weightedCount_ = 0;
    state.exponentialMovingAverageState_.weightedSum_   = 0;

    KeyValueTreeBuilder kvtBuilder;
    const std::string   identifier = "test-module";
    state.writeState(kvtBuilder.rootObject(), identifier);
    KeyValueTreeObject stateStoredInKvt = kvtBuilder.build();

    // invalidate state
    state.adaptiveForceConstantScale_                   = -1;
    state.stepsSinceLastCalculation_                    = -1;
    state.exponentialMovingAverageState_.increasing_    = true;
    state.exponentialMovingAverageState_.weightedCount_ = -1;
    state.exponentialMovingAverageState_.weightedSum_   = -1;

    // read back the original state
    state.readState(stateStoredInKvt, identifier);

    EXPECT_EQ(state.adaptiveForceConstantScale_, 1.0);
    EXPECT_EQ(state.stepsSinceLastCalculation_, 0);

    EXPECT_EQ(state.exponentialMovingAverageState_.increasing_, false);
    EXPECT_EQ(state.exponentialMovingAverageState_.weightedCount_, 0);
    EXPECT_EQ(state.exponentialMovingAverageState_.weightedSum_, 0);
}


} // namespace gmx
