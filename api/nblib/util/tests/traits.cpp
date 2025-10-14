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
 * This implements test for nblib general purpose traits
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */

#include "nblib/util/traits.hpp"

#include "testhelpers.h"

namespace nblib
{

TEST(NblibTraitsUtils, FuseTwo)
{
    using TL1 = TypeList<float, int>;
    using TL2 = TypeList<double, unsigned>;

    using TL_fused = FuseTwo<TL1, TL2>;
    using TL_ref   = TypeList<float, int, double, unsigned>;

    constexpr bool match = std::is_same_v<TL_fused, TL_ref>;
    EXPECT_TRUE(match);
}

TEST(NblibTraitsUtils, Fuse)
{
    using TL1 = TypeList<float, int>;
    using TL2 = TypeList<double, unsigned>;
    using TL3 = TypeList<char, short>;

    using TL_fused = Fuse<TL1, TL2, TL3>;
    using TL_ref   = TypeList<float, int, double, unsigned, char, short>;

    constexpr bool match = std::is_same_v<TL_fused, TL_ref>;
    EXPECT_TRUE(match);
}

TEST(NblibTraitsUtils, Repeat)
{
    using TL1 = TypeList<float, int>;

    using TL_repeated = Repeat<TL1, 3>;
    using TL_ref      = TypeList<float, int, float, int, float, int>;

    constexpr bool match = std::is_same_v<TL_repeated, TL_ref>;
    EXPECT_TRUE(match);
}

TEST(NblibTraitsUtils, FindIndexTuple1)
{
    using TupleType = std::tuple<float>;

    constexpr int floatIndex = FindIndex<float, TupleType>{};

    constexpr int outOfRange = FindIndex<unsigned, TupleType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(1, outOfRange);
}

TEST(NblibTraitsUtils, FindIndexTuple2)
{
    using TupleType = std::tuple<float, int>;

    constexpr int floatIndex = FindIndex<float, TupleType>{};
    constexpr int intIndex   = FindIndex<int, TupleType>{};

    constexpr int outOfRange = FindIndex<unsigned, TupleType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(1, intIndex);
    EXPECT_EQ(2, outOfRange);
}

TEST(NblibTraitsUtils, FindIndexTypeList1)
{
    using ListType = TypeList<float>;

    constexpr int floatIndex = FindIndex<float, ListType>{};

    constexpr int outOfRange = FindIndex<unsigned, ListType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(1, outOfRange);
}

TEST(NblibTraitsUtils, FindIndexTypeList2)
{
    using ListType = TypeList<float, int>;

    constexpr int floatIndex = FindIndex<float, ListType>{};
    constexpr int intIndex   = FindIndex<int, ListType>{};

    constexpr int outOfRange = FindIndex<unsigned, ListType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(1, intIndex);
    EXPECT_EQ(2, outOfRange);
}


TEST(NblibTraitsUtils, Contains)
{
    using ListType = TypeList<float, int>;

    constexpr bool hasFloat = Contains<float, ListType>{};
    constexpr bool hasInt   = Contains<int, ListType>{};
    constexpr bool hasUint  = Contains<unsigned, ListType>{};

    EXPECT_TRUE(hasFloat);
    EXPECT_TRUE(hasInt);
    EXPECT_FALSE(hasUint);
}

TEST(NblibTraitsUtils, FindIndexTupleRepeated)
{
    using TupleType = std::tuple<float, float, int>;

    constexpr int floatIndex = FindIndex<float, TupleType>{};

    constexpr int intIndex = FindIndex<int, TupleType>{};

    constexpr int outOfRange = FindIndex<unsigned, TupleType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(2, intIndex);
    EXPECT_EQ(3, outOfRange);
}

TEST(NblibTraitsUtils, FindIndexTypeListRepeated)
{
    using TupleType = TypeList<float, float, int>;

    constexpr int floatIndex = FindIndex<float, TupleType>{};

    constexpr int intIndex = FindIndex<int, TupleType>{};

    constexpr int outOfRange = FindIndex<unsigned, TupleType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(2, intIndex);
    EXPECT_EQ(3, outOfRange);
}


} // namespace nblib
