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
 * This implements basic nblib box tests
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#include "nblib/listed_forces/bondtypes.h"

#include "testutils/testasserts.h"

#include "nblib/util/util.hpp"

namespace nblib
{

namespace test_detail
{

template<class B>
void testTwoParameterBondEquality([[maybe_unused]] const B& deduceType)
{
    B a(1, 2);
    B b(1, 2);
    EXPECT_TRUE(a == b);

    B c(1, 3);
    EXPECT_FALSE(a == c);
}

template<class B>
void testThreeParameterBondEquality([[maybe_unused]] const B& deduceType)
{
    B a(1, 2, 3);
    B b(1, 2, 3);
    EXPECT_TRUE(a == b);

    B c(2, 3, 4);
    EXPECT_FALSE(a == c);
}

template<class B>
void testTwoParameterBondLessThan([[maybe_unused]] const B& deduceType)
{
    B a(1, 2);
    B b(1, 3);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);

    B c(1, 2);
    B d(1, 2);
    EXPECT_FALSE(c < d);

    B e(2, 1);
    B f(3, 1);
    EXPECT_TRUE(e < f);
    EXPECT_FALSE(f < e);
}

template<class B>
void testThreeParameterBondLessThan([[maybe_unused]] const B& deduceType)
{
    B a(1, 2, 1);
    B b(1, 3, 1);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);

    B c(1, 2, 3);
    B d(1, 2, 3);
    EXPECT_FALSE(c < d);

    B e(4, 1, 3);
    B f(5, 1, 2);
    EXPECT_TRUE(e < f);
    EXPECT_FALSE(f < e);
}

} // namespace test_detail

TEST(NBlibTest, BondTypesOperatorEqualWorks)
{
    auto bondList3 = std::make_tuple(
            HarmonicBondType(), G96BondType(), FENEBondType(), HalfAttractiveQuarticBondType());
    for_each_tuple([](const auto& b) { test_detail::testTwoParameterBondEquality(b); }, bondList3);

    auto bondList4 = std::make_tuple(CubicBondType(), MorseBondType());
    for_each_tuple([](const auto& b) { test_detail::testThreeParameterBondEquality(b); }, bondList4);
}

TEST(NBlibTest, BondTypesLessThanWorks)
{
    auto bondList3 = std::make_tuple(
            HarmonicBondType(), G96BondType(), FENEBondType(), HalfAttractiveQuarticBondType());
    for_each_tuple([](const auto& b) { test_detail::testTwoParameterBondLessThan(b); }, bondList3);

    auto bondList4 = std::make_tuple(CubicBondType(), MorseBondType());
    for_each_tuple([](const auto& b) { test_detail::testThreeParameterBondLessThan(b); }, bondList4);
}


} // namespace nblib
