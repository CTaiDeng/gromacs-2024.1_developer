/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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
 * \brief Testing gmx::accessor_policy.
 *
 * \author Christian Blau <cblau@gwdg.de>
 */
#include "gmxpre.h"

#include "gromacs/mdspan/accessor_policy.h"

#include <array>

#include <gtest/gtest.h>

namespace gmx
{

class BasicAccessorPolicy : public ::testing::Test
{
public:
    std::array<float, 3>  testdata = { { 1., 2., 3. } };
    accessor_basic<float> acc;
};

TEST_F(BasicAccessorPolicy, Decay)
{
    EXPECT_EQ(acc.decay(testdata.data()), testdata.data());
}

TEST_F(BasicAccessorPolicy, Access)
{
    for (size_t i = 0; i < testdata.size(); ++i)
    {
        EXPECT_EQ(acc.access(testdata.data(), i), testdata[i]);
    }
}

TEST_F(BasicAccessorPolicy, Offset)
{
    for (size_t i = 0; i < testdata.size(); ++i)
    {
        EXPECT_EQ(acc.offset(testdata.data(), i), testdata.data() + i);
    }
}

TEST_F(BasicAccessorPolicy, CopyAccessor)
{
    const auto newAcc = acc;

    EXPECT_EQ(acc.decay(testdata.data()), newAcc.decay(testdata.data()));
    for (size_t i = 0; i < testdata.size(); ++i)
    {
        EXPECT_EQ(acc.access(testdata.data(), i), newAcc.access(testdata.data(), i));
    }

    for (size_t i = 0; i < testdata.size(); ++i)
    {
        EXPECT_EQ(acc.offset(testdata.data(), i), newAcc.offset(testdata.data(), i));
    }
}

} // namespace gmx
