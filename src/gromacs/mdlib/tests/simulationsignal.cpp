/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * Tests for the mdrun signalling functionality
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "gromacs/mdlib/simulationsignal.h"

#include <gtest/gtest.h>

#include "gromacs/utility/arrayref.h"

namespace gmx
{

namespace test
{

//! Check that a null signaller can be called without problems
TEST(NullSignalTest, NullSignallerWorks)
{
    SimulationSignaller signaller(nullptr, nullptr, nullptr, false, false);
    EXPECT_EQ(0, signaller.getCommunicationBuffer().size());
    signaller.finalizeSignals();
}

//! Test fixture for mdrun signalling
class SignalTest : public ::testing::Test
{
public:
    SignalTest() : signals_{}
    {
        signals_[0].sig = 1;
        signals_[1].sig = -1;
    }
    //! Default object to hold signals
    SimulationSignals signals_;
};

TEST_F(SignalTest, NoSignalPropagatesIfNoSignallingTakesPlace)
{
    SimulationSignaller signaller(&signals_, nullptr, nullptr, false, false);
    EXPECT_EQ(0, signaller.getCommunicationBuffer().size());
    signaller.finalizeSignals();
    EXPECT_EQ(1, signals_[0].sig);
    EXPECT_EQ(-1, signals_[1].sig);
    EXPECT_EQ(0, signals_[2].sig);
    EXPECT_EQ(0, signals_[0].set);
    EXPECT_EQ(0, signals_[1].set);
    EXPECT_EQ(0, signals_[2].set);
}

TEST_F(SignalTest, LocalIntraSimSignalPropagatesWhenIntraSimSignalTakesPlace)
{
    SimulationSignaller signaller(&signals_, nullptr, nullptr, false, true);
    EXPECT_NE(0, signaller.getCommunicationBuffer().size());
    signaller.finalizeSignals();
    EXPECT_EQ(0, signals_[0].sig);
    EXPECT_EQ(0, signals_[1].sig);
    EXPECT_EQ(0, signals_[2].sig);
    EXPECT_EQ(1, signals_[0].set);
    EXPECT_EQ(-1, signals_[1].set);
    EXPECT_EQ(0, signals_[2].set);
}

TEST_F(SignalTest, LocalIntraSimSignalPropagatesWhenInterSimTakesPlace)
{
    SimulationSignaller signaller(&signals_, nullptr, nullptr, true, false);
    EXPECT_NE(0, signaller.getCommunicationBuffer().size());
    // Can't call finalizeSignals without a full commrec
    signaller.setSignals();
    EXPECT_EQ(0, signals_[0].sig);
    EXPECT_EQ(0, signals_[1].sig);
    EXPECT_EQ(0, signals_[2].sig);
    EXPECT_EQ(1, signals_[0].set);
    EXPECT_EQ(-1, signals_[1].set);
    EXPECT_EQ(0, signals_[2].set);
}

TEST_F(SignalTest, LocalIntraSimSignalPropagatesWhenBothTakePlace)
{
    SimulationSignaller signaller(&signals_, nullptr, nullptr, true, true);
    EXPECT_NE(0, signaller.getCommunicationBuffer().size());
    // Can't call finalizeSignals without a full commrec
    signaller.setSignals();
    EXPECT_EQ(0, signals_[0].sig);
    EXPECT_EQ(0, signals_[1].sig);
    EXPECT_EQ(0, signals_[2].sig);
    EXPECT_EQ(1, signals_[0].set);
    EXPECT_EQ(-1, signals_[1].set);
    EXPECT_EQ(0, signals_[2].set);
}

TEST_F(SignalTest, NonLocalSignalDoesntPropagateWhenIntraSimSignalTakesPlace)
{
    signals_[0].isLocal = false;
    SimulationSignaller signaller(&signals_, nullptr, nullptr, false, true);
    EXPECT_NE(0, signaller.getCommunicationBuffer().size());
    signaller.finalizeSignals();
    EXPECT_EQ(1, signals_[0].sig);
    EXPECT_EQ(0, signals_[1].sig);
    EXPECT_EQ(0, signals_[2].sig);
    EXPECT_EQ(0, signals_[0].set);
    EXPECT_EQ(-1, signals_[1].set);
    EXPECT_EQ(0, signals_[2].set);
}

TEST_F(SignalTest, NonLocalSignalPropagatesWhenInterSimSignalTakesPlace)
{
    signals_[0].isLocal = false;
    SimulationSignaller signaller(&signals_, nullptr, nullptr, true, false);
    EXPECT_NE(0, signaller.getCommunicationBuffer().size());
    // Can't call finalizeSignals without a full commrec
    signaller.setSignals();
    EXPECT_EQ(0, signals_[0].sig);
    EXPECT_EQ(0, signals_[1].sig);
    EXPECT_EQ(0, signals_[2].sig);
    EXPECT_EQ(1, signals_[0].set);
    EXPECT_EQ(-1, signals_[1].set);
    EXPECT_EQ(0, signals_[2].set);
}

TEST_F(SignalTest, NonLocalSignalPropagatesWhenBothTakePlace)
{
    signals_[0].isLocal = false;
    SimulationSignaller signaller(&signals_, nullptr, nullptr, true, true);
    EXPECT_NE(0, signaller.getCommunicationBuffer().size());
    // Can't call finalizeSignals without a full commrec
    signaller.setSignals();
    EXPECT_EQ(0, signals_[0].sig);
    EXPECT_EQ(0, signals_[1].sig);
    EXPECT_EQ(0, signals_[2].sig);
    EXPECT_EQ(1, signals_[0].set);
    EXPECT_EQ(-1, signals_[1].set);
    EXPECT_EQ(0, signals_[2].set);
}

} // namespace test
} // namespace gmx
