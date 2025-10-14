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
 * \brief Tests for the EnergyDriftTacker class
 *
 * \author berk Hess <hess@kth.se>
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "gromacs/mdlib/energydrifttracker.h"

#include <gtest/gtest.h>

#include "testutils/testasserts.h"

namespace gmx
{

namespace
{

TEST(EnergyDriftTracker, emptyWorks)
{
    EnergyDriftTracker tracker(1);

    EXPECT_EQ(tracker.timeInterval(), 0);
    EXPECT_EQ(tracker.energyDrift(), 0);
}

TEST(EnergyDriftTracker, onePointWorks)
{
    EnergyDriftTracker tracker(1);

    tracker.addPoint(1.5, -3.5_real);
    EXPECT_EQ(tracker.timeInterval(), 0);
    EXPECT_EQ(tracker.energyDrift(), 0);
}

TEST(EnergyDriftTracker, manyPointsWorks)
{
    EnergyDriftTracker tracker(10);

    tracker.addPoint(1.5, 2.5_real);
    tracker.addPoint(3.5, 4.0_real);
    tracker.addPoint(5.5, -5.5_real);
    EXPECT_FLOAT_EQ(tracker.timeInterval(), 4.0_real);
    EXPECT_FLOAT_EQ(tracker.energyDrift(), -0.2_real);
}

} // namespace

} // namespace gmx
