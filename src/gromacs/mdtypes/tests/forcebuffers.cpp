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
 * \brief
 * Tests for the ForceBuffers and ForceBuffersView classes.
 *
 * \author berk Hess <hess@kth.se>
 * \ingroup module_mdtypes
 */
#include "gmxpre.h"

#include "gromacs/mdtypes/forcebuffers.h"

#include <array>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "testutils/testasserts.h"

namespace gmx
{

const std::array<RVec, 2> c_forces = { { { 0.5, 0.1, 1.2 }, { -2.1, 0.2, 0.3 } } };

TEST(ForceBuffers, ConstructsUnpinned)
{
    ForceBuffers forceBuffers;

    EXPECT_EQ(forceBuffers.pinningPolicy(), PinningPolicy::CannotBePinned);
}

TEST(ForceBuffers, ConstructsPinned)
{
    ForceBuffers forceBuffers(false, PinningPolicy::PinnedIfSupported);

    EXPECT_EQ(forceBuffers.pinningPolicy(), PinningPolicy::PinnedIfSupported);
}

TEST(ForceBuffers, ConstructsEmpty)
{
    ForceBuffers forceBuffers;

    EXPECT_EQ(forceBuffers.view().force().size(), 0);
}

TEST(ForceBuffers, ResizeWorks)
{
    ForceBuffers forceBuffers;

    forceBuffers.resize(2);
    EXPECT_EQ(forceBuffers.view().force().size(), 2);
}

TEST(ForceBuffers, PaddingWorks)
{
    ForceBuffers forceBuffers;

    forceBuffers.resize(2);
    auto paddedRef = forceBuffers.view().forceWithPadding();
    EXPECT_EQ(paddedRef.unpaddedArrayRef().size(), 2);
    EXPECT_GT(paddedRef.size(), 2);
}


TEST(ForceBuffers, CopyWorks)
{
    ForceBuffers forceBuffers;

    forceBuffers.resize(2);
    auto  force = forceBuffers.view().force();
    Index i     = 0;
    for (RVec& v : force)
    {
        v = c_forces[i];
        i++;
    }

    ForceBuffers forceBuffersCopy;
    forceBuffersCopy = forceBuffers;
    auto forceCopy   = forceBuffersCopy.view().force();
    EXPECT_EQ(forceBuffersCopy.view().force().size(), 2);
    for (Index i = 0; i < ssize(forceCopy); i++)
    {
        for (int d = 0; d < DIM; d++)
        {
            EXPECT_EQ(forceCopy[i][d], force[i][d]);
        }
    }
}

TEST(ForceBuffers, CopyDoesNotPin)
{
    ForceBuffers forceBuffers(false, PinningPolicy::PinnedIfSupported);

    ForceBuffers forceBuffersCopy;
    forceBuffersCopy = forceBuffers;
    EXPECT_EQ(forceBuffersCopy.pinningPolicy(), PinningPolicy::CannotBePinned);
}

} // namespace gmx
