/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2022- The GROMACS Authors
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
 *
 * \brief Tests routines in booltype.h .
 *
 * \author Christian Blau <blau@kth.se>
 */

#include "gmxpre.h"

#include "gromacs/utility/booltype.h"

#include <vector>

#include <gtest/gtest.h>

#include "gromacs/utility/arrayref.h"

namespace gmx
{
namespace test
{
namespace
{

TEST(BoolType, ImplicitConversion)
{
    BoolType variable(true);
    EXPECT_TRUE(variable);
}

TEST(BoolType, FalseByDefault)
{
    BoolType variable;
    EXPECT_FALSE(variable);
}

TEST(BoolType, Assignment)
{
    BoolType variable;
    variable = true;
    EXPECT_TRUE(variable);
}

TEST(BoolType, Copy)
{
    BoolType variable = true;
    BoolType another  = false;
    variable          = another;
    EXPECT_FALSE(variable);
}

// being able to create an ArrayRef on a std::vector of this type is
// the main reason for its existence, so we test it here
TEST(BoolType, ArrayRefCanBeCreated)
{
    std::vector<BoolType> vecOfBoolType(3);
    ArrayRef<BoolType>    view(vecOfBoolType);
    vecOfBoolType[1] = true;
    EXPECT_FALSE(view[0]);
    EXPECT_TRUE(view[1]);
    EXPECT_FALSE(view[2]);
}

TEST(BoolType, CanBeCastToBool)
{
    BoolType val    = true;
    bool*    valPtr = reinterpret_cast<bool*>(&val);
    EXPECT_TRUE(*valPtr);
    val = false;
    EXPECT_FALSE(val);
    EXPECT_FALSE(*valPtr);
    *valPtr = true;
    EXPECT_TRUE(val);
    EXPECT_TRUE(*valPtr);
}

TEST(BoolType, HasSizeOfBool)
{
    EXPECT_EQ(sizeof(BoolType), sizeof(bool));
}

TEST(BoolType, HasAlignmentOfBool)
{
    EXPECT_EQ(alignof(BoolType), alignof(bool));
}

TEST(ArrayRefFromBoolTypeVector, CanConstructEmpty)
{
    std::vector<BoolType> boolVector;
    ArrayRef<bool>        boolRef = makeArrayRef(boolVector);
    EXPECT_EQ(0U, boolRef.size());
    EXPECT_TRUE(boolRef.empty());
}

TEST(ArrayRefFromBoolTypeVector, Works)
{
    std::vector<BoolType> boolVector = { true, false, true };
    ArrayRef<bool>        boolRef    = makeArrayRef(boolVector);
    EXPECT_EQ(3U, boolRef.size());
    EXPECT_TRUE(boolRef[0]);
    EXPECT_FALSE(boolRef[1]);
    EXPECT_TRUE(boolRef[2]);
}

TEST(ArrayRefFromBoolTypeVector, CanConstructConstEmpty)
{
    std::vector<BoolType> boolVector;
    ArrayRef<const bool>  boolRef = makeArrayRef(boolVector);
    EXPECT_EQ(0U, boolRef.size());
    EXPECT_TRUE(boolRef.empty());
}

TEST(ArrayRefFromBoolTypeVector, ConstWorks)
{
    std::vector<BoolType> boolVector = { true, false, true };
    ArrayRef<const bool>  boolRef    = makeArrayRef(boolVector);
    EXPECT_EQ(3U, boolRef.size());
    EXPECT_TRUE(boolRef[0]);
    EXPECT_FALSE(boolRef[1]);
    EXPECT_TRUE(boolRef[2]);
}


} // namespace
} // namespace test
} // namespace gmx
