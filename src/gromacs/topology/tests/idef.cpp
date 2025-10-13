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
 * Implements test of InteractionList routines
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_topology
 */
#include "gmxpre.h"

#include "gromacs/topology/idef.h"

#include <array>

#include <gtest/gtest.h>

namespace gmx
{
namespace
{

TEST(InteractionListTest, EmptyWorks)
{
    InteractionList ilist;
    EXPECT_TRUE(ilist.empty());
    EXPECT_EQ(ilist.size(), 0);
}

TEST(InteractionListTest, CanAddInteractionArray)
{
    InteractionList    ilist;
    int                parameterType  = 0;
    std::array<int, 1> singleAtomList = { 1 };
    ilist.push_back(parameterType, singleAtomList);
    EXPECT_FALSE(ilist.empty());
    EXPECT_EQ(ilist.size(), 2);
    EXPECT_EQ(ilist.iatoms[0], parameterType);
    EXPECT_EQ(ilist.iatoms[1], 1);
}

TEST(InteractionListTest, CanAddInteractionArrayMultipleAtoms)
{
    InteractionList    ilist;
    int                parameterType = 0;
    std::array<int, 3> atomList      = { 1, 2, 3 };
    ilist.push_back(parameterType, atomList);
    EXPECT_FALSE(ilist.empty());
    EXPECT_EQ(ilist.size(), 4);
    EXPECT_EQ(ilist.iatoms[0], parameterType);
    EXPECT_EQ(ilist.iatoms[1], 1);
    EXPECT_EQ(ilist.iatoms[2], 2);
    EXPECT_EQ(ilist.iatoms[3], 3);
}

TEST(InteractionListTest, CanAddInteractionPointer)
{
    InteractionList    ilist;
    int                parameterType  = 0;
    std::array<int, 1> singleAtomList = { 1 };
    ilist.push_back(parameterType, singleAtomList.size(), singleAtomList.data());
    EXPECT_FALSE(ilist.empty());
    EXPECT_EQ(ilist.size(), 2);
    EXPECT_EQ(ilist.iatoms[0], parameterType);
    EXPECT_EQ(ilist.iatoms[1], 1);
}

TEST(InteractionListTest, CanAddListToOtherList)
{
    InteractionList firstList;
    int             firstParameterType = 0;
    {
        std::array<int, 1> singleAtomList = { 1 };
        firstList.push_back(firstParameterType, singleAtomList);
        EXPECT_FALSE(firstList.empty());
        EXPECT_EQ(firstList.size(), 2);
        EXPECT_EQ(firstList.iatoms[0], firstParameterType);
        EXPECT_EQ(firstList.iatoms[1], 1);
    }
    InteractionList secondList;
    int             secondParameterType = 1;
    {
        std::array<int, 3> atomList = { 1, 2, 3 };
        secondList.push_back(secondParameterType, atomList);
        EXPECT_FALSE(secondList.empty());
        EXPECT_EQ(secondList.size(), 4);
        EXPECT_EQ(secondList.iatoms[0], secondParameterType);
        EXPECT_EQ(secondList.iatoms[1], 1);
        EXPECT_EQ(secondList.iatoms[2], 2);
        EXPECT_EQ(secondList.iatoms[3], 3);
    }
    firstList.append(secondList);
    EXPECT_EQ(firstList.size(), 6);
    EXPECT_EQ(firstList.iatoms[2], secondParameterType);
    EXPECT_EQ(firstList.iatoms[3], 1);
    EXPECT_EQ(firstList.iatoms[4], 2);
    EXPECT_EQ(firstList.iatoms[5], 3);
}

TEST(InteractionListTest, ClearingWorks)
{
    InteractionList    ilist;
    int                parameterType  = 0;
    std::array<int, 1> singleAtomList = { 1 };
    ilist.push_back(parameterType, singleAtomList);
    EXPECT_FALSE(ilist.empty());
    EXPECT_EQ(ilist.size(), 2);
    EXPECT_EQ(ilist.iatoms[0], parameterType);
    EXPECT_EQ(ilist.iatoms[1], 1);
    ilist.clear();
    EXPECT_TRUE(ilist.empty());
    EXPECT_EQ(ilist.size(), 0);
}

} // namespace

} // namespace gmx
