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
 * \brief Tests for gmx::DefaultInitializationAllocator used in std::vector
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_utility
 */

#include "gmxpre.h"

#include "gromacs/utility/defaultinitializationallocator.h"

#include <gtest/gtest.h>

#include "gromacs/utility/gmxassert.h"


namespace gmx
{
namespace test
{
namespace
{

TEST(DefaultInitializationAllocator, PerformsValueInitialization)
{
    std::vector<int, DefaultInitializationAllocator<int>> v;

    v.resize(1, 2);
    EXPECT_EQ(v[0], 2);
}

TEST(DefaultInitializationAllocator, PerformsNoInitialization)
{
    std::vector<int, DefaultInitializationAllocator<int>> v{ 1, 2, 3 };

    const int* oldData = v.data();
    v.resize(0);
    v.resize(3);
    GMX_RELEASE_ASSERT(v.data() == oldData,
                       "According to the C++ standard std::vector will not reallocate when the "
                       "capacity is sufficient");
    // The allocation is the same, so the default initialization should
    // not have changed the contents
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);
}

} // namespace
} // namespace test
} // namespace gmx
