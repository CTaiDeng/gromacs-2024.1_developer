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
 * \brief
 * Test routines that handle topology directive data structures
 * and files.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 */
#include "gmxpre.h"

#include "gromacs/gmxpreprocess/topdirs.h"

#include <gtest/gtest.h>

#include "gromacs/utility/enumerationhelpers.h"

TEST(TopDirTests, NamesArrayHasCorrectSize)
{
    for (auto d : gmx::EnumerationWrapper<Directive>())
    {
        // If the enumeration is extended, but there is no matching
        // name, then at least one element will be value initialized,
        // ie. to nullptr, which this test will catch.
        const auto* name = enumValueToString(d);
        EXPECT_NE(name, nullptr);
    }
}
