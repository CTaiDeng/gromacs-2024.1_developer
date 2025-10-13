/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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
 * Implements low level test of manyautocorrelation routines
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_correlationfunctions
 */
#include "gmxpre.h"

#include "gromacs/correlationfunctions/manyautocorrelation.h"

#include <cmath>

#include <memory>

#include <gtest/gtest.h>

#include "gromacs/utility/exceptions.h"

#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace
{

class ManyAutocorrelationTest : public ::testing::Test
{
};

TEST_F(ManyAutocorrelationTest, Empty)
{
    std::vector<std::vector<real>> c;
    EXPECT_THROW_GMX(many_auto_correl(&c), gmx::InconsistentInputError);
}

#ifndef NDEBUG
TEST_F(ManyAutocorrelationTest, DifferentLength)
{
    std::vector<std::vector<real>> c;
    c.resize(3);
    c[0].resize(10);
    c[1].resize(10);
    c[2].resize(8);
    EXPECT_THROW_GMX(many_auto_correl(&c), gmx::InconsistentInputError);
}
#endif

} // namespace

} // namespace gmx
