/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * \brief Tests for gmx::AlignedAllocator and gmx::PageAlignedAllocator.
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_utility
 */

#include "gmxpre.h"

#include "gromacs/utility/alignedallocator.h"

#include <vector>

#include <gtest/gtest.h>

#include "gromacs/math/vectypes.h"

#ifndef DOXYGEN
namespace gmx
{
namespace test
{

//! Declare allocator types to test.
using AllocatorTypesToTest = ::testing::Types<AlignedAllocator<real>,
                                              PageAlignedAllocator<real>,
                                              AlignedAllocator<int>,
                                              PageAlignedAllocator<int>,
                                              AlignedAllocator<RVec>,
                                              PageAlignedAllocator<RVec>>;

TYPED_TEST_SUITE(AllocatorTest, AllocatorTypesToTest);

} // namespace test
} // namespace gmx

#endif

// Includes tests common to all allocation policies.
#include "gromacs/utility/tests/alignedallocator_impl.h"

namespace gmx
{
namespace test
{

TYPED_TEST(AllocatorTest, StatelessAllocatorUsesNoMemory)
{
    using value_type = typename TypeParam::value_type;
    EXPECT_EQ(sizeof(std::vector<value_type>), sizeof(std::vector<value_type, TypeParam>));
}

TEST(AllocatorUntypedTest, Comparison)
{
    // Should always be true for the same policy, indpendent of value_type
    EXPECT_EQ(AlignedAllocator<float>{}, AlignedAllocator<double>{});
    EXPECT_EQ(PageAlignedAllocator<float>{}, PageAlignedAllocator<double>{});
}

} // namespace test
} // namespace gmx
