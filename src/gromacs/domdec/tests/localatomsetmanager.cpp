/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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
 * Tests for general functionality in gmx::LocalAtomSetManager and
 * gmx::LocalAtomSet, which is only accesible through the manager.
 *
 * TODO: add testing for behaviour on multiple ranks once gmx_ga2la_t
 * may be set up individually and outside domain decomposition initialisation.
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \ingroup module_domdec
 */
#include "gmxpre.h"

#include "gromacs/domdec/localatomsetmanager.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"

#include "testutils/testasserts.h"

namespace gmx
{

extern template LocalAtomSet LocalAtomSetManager::add<void, void>(ArrayRef<const int> globalAtomIndex);

namespace test
{

TEST(LocalAtomSetManager, CanAddEmptyLocalAtomSet)
{
    LocalAtomSetManager    manager;
    const std::vector<int> emptyIndex = {};
    LocalAtomSet           emptyGroup(manager.add(emptyIndex));
    const std::vector<int> globalIndexFromGroup(emptyGroup.globalIndex().begin(),
                                                emptyGroup.globalIndex().end());
    ASSERT_THAT(globalIndexFromGroup, testing::ContainerEq(emptyIndex));
}

TEST(LocalAtomSetManager, CanAddandReadLocalAtomSetIndices)
{
    LocalAtomSetManager manager;

    const std::vector<int> index = { 5, 10 };
    LocalAtomSet           newGroup(manager.add(index));
    std::vector<int>       readIndex;
    for (const auto& i : newGroup.localIndex())
    {
        readIndex.push_back(i);
    }

    ASSERT_THAT(readIndex, testing::ContainerEq(index));
}

} // namespace test
} // namespace gmx
