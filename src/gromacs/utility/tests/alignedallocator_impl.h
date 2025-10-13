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

/*! \libinternal \file
 * \brief Tests for allocators that offer a minimum alignment.
 *
 * This implementation header can be included in multiple modules
 * tests, which is currently needed because gpu_utils is physically
 * separate from the utility module.
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_TESTS_ALIGNEDALLOCATOR_IMPL_H
#define GMX_UTILITY_TESTS_ALIGNEDALLOCATOR_IMPL_H

#include <cstddef>

#include <vector>

#include <gtest/gtest.h>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"

namespace gmx
{
namespace test
{

/*! \libinternal
 * \brief Templated test fixture. */
template<typename T>
class AllocatorTest : public ::testing::Test
{
public:
    /*! \brief Return a bitmask for testing the alignment.
     *
     * e.g. for 128-byte alignment the mask is 128-1 - all of
     * these bits should be zero in pointers that have the
     * intended alignment. */
    std::size_t mask(const T& allocator) { return allocator.alignment() - 1; }
};

// NB need to use this->mask() because of GoogleTest quirks

TYPED_TEST(AllocatorTest, AllocatorAlignAllocatesWithAlignment) //NOLINT(misc-definitions-in-headers)
{
    using pointer = typename TypeParam::value_type*;
    TypeParam a;
    pointer   p = a.allocate(1000);

    EXPECT_EQ(0, reinterpret_cast<std::size_t>(p) & this->mask(a));
    a.deallocate(p, 1000);
}


TYPED_TEST(AllocatorTest, VectorAllocatesAndResizesWithAlignment) //NOLINT(misc-definitions-in-headers)
{
    using value_type = typename TypeParam::value_type;
    std::vector<value_type, TypeParam> v(10);
    EXPECT_EQ(0, reinterpret_cast<std::size_t>(v.data()) & this->mask(v.get_allocator()));

    // Reserve a few times to check things work ok, making sure we
    // will trigger several reallocations on common vector
    // implementations.
    for (std::size_t i = 1000; i <= 10000; i += 1000)
    {
        v.resize(i);
        EXPECT_EQ(0, reinterpret_cast<std::size_t>(v.data()) & this->mask(v.get_allocator()));
    }
}

TYPED_TEST(AllocatorTest, VectorAllocatesAndReservesWithAlignment) //NOLINT(misc-definitions-in-headers)
{
    using value_type = typename TypeParam::value_type;
    std::vector<value_type, TypeParam> v(10);
    EXPECT_EQ(0, reinterpret_cast<std::size_t>(v.data()) & this->mask(v.get_allocator()));

    // Reserve a few times to check things work ok, making sure we
    // will trigger several reallocations on common vector
    // implementations.
    for (std::size_t i = 1000; i <= 10000; i += 1000)
    {
        v.reserve(i);
        EXPECT_EQ(0, reinterpret_cast<std::size_t>(v.data()) & this->mask(v.get_allocator()));
    }
}

TYPED_TEST(AllocatorTest, Move) //NOLINT(misc-definitions-in-headers)
{
    using value_type = typename TypeParam::value_type;
    std::vector<value_type, TypeParam> v1(1);
    value_type*                        data = v1.data();
    EXPECT_NE(data, nullptr);
    std::vector<value_type, TypeParam> v2(std::move(v1));
    EXPECT_EQ(data, v2.data());
}

} // namespace test
} // namespace gmx

#endif
