/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * \brief Tests for random seed functions
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \ingroup module_random
 */
#include "gmxpre.h"

#include "gromacs/random/seed.h"

#include <gtest/gtest.h>

namespace gmx
{

namespace
{

// Test the random device call
TEST(SeedTest, makeRandomSeed)
{
    // Unlike Sony, we do not use "4" as a constant random value, so the only
    // thing we can check for the random device is that multiple calls to
    // it produce different results.
    // We choose to ignore the 2^-64 probability this will happen by chance;
    // if you execute the unit tests once per second you might have to run them
    // an extra time rougly once per 300 billion years - apologies in advance!

    uint64_t i0 = makeRandomSeed();
    uint64_t i1 = makeRandomSeed();

    EXPECT_NE(i0, i1);
}

} // namespace

} // namespace gmx
