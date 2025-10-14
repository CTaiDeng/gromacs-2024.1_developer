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
 * Tests amplitude lookup for density fitting
 *
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gromacs/applied_forces/densityfitting/densityfittingamplitudelookup.h"

#include <vector>

#include <gtest/gtest.h>

#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/utility/arrayref.h"

namespace gmx
{

class DensityFittingAmplitudeLookupTest : public ::testing::Test
{
protected:
    std::vector<real> masses_        = { 2, 3, 4 };
    std::vector<real> charges_       = { 20, 30, 40 };
    t_mdatoms         atoms_         = {};
    std::vector<int>  lookupIndices_ = { 1, 2 };
};

TEST_F(DensityFittingAmplitudeLookupTest, Unity)
{
    DensityFittingAmplitudeLookup lookup(DensityFittingAmplitudeMethod::Unity);
    const auto                    lookupResult = lookup(charges_, masses_, lookupIndices_);
    EXPECT_EQ(lookupResult[0], 1);
    EXPECT_EQ(lookupResult[1], 1);
}

TEST_F(DensityFittingAmplitudeLookupTest, Charge)
{
    DensityFittingAmplitudeLookup lookup(DensityFittingAmplitudeMethod::Charge);
    const auto                    lookupResult = lookup(charges_, masses_, lookupIndices_);
    EXPECT_EQ(lookupResult[0], 30);
    EXPECT_EQ(lookupResult[1], 40);
}

TEST_F(DensityFittingAmplitudeLookupTest, Masses)
{
    DensityFittingAmplitudeLookup lookup(DensityFittingAmplitudeMethod::Mass);
    const auto                    lookupResult = lookup(charges_, masses_, lookupIndices_);
    EXPECT_EQ(lookupResult[0], 3);
    EXPECT_EQ(lookupResult[1], 4);
}

TEST_F(DensityFittingAmplitudeLookupTest, CanCopyAssign)
{
    DensityFittingAmplitudeLookup lookup(DensityFittingAmplitudeMethod::Unity);
    DensityFittingAmplitudeLookup lookupCopied = lookup;
    const auto                    lookupResult = lookupCopied(charges_, masses_, lookupIndices_);
    EXPECT_EQ(lookupResult[0], 1);
    EXPECT_EQ(lookupResult[1], 1);
}

TEST_F(DensityFittingAmplitudeLookupTest, CanCopyConstruct)
{
    DensityFittingAmplitudeLookup lookup(DensityFittingAmplitudeMethod::Unity);
    DensityFittingAmplitudeLookup lookupCopied(lookup);
    const auto                    lookupResult = lookupCopied(charges_, masses_, lookupIndices_);
    EXPECT_EQ(lookupResult[0], 1);
    EXPECT_EQ(lookupResult[1], 1);
}

TEST_F(DensityFittingAmplitudeLookupTest, CanMoveAssign)
{
    DensityFittingAmplitudeLookup lookup(DensityFittingAmplitudeMethod::Unity);
    DensityFittingAmplitudeLookup lookupCopied = std::move(lookup);
    const auto                    lookupResult = lookupCopied(charges_, masses_, lookupIndices_);
    EXPECT_EQ(lookupResult[0], 1);
    EXPECT_EQ(lookupResult[1], 1);
}

TEST_F(DensityFittingAmplitudeLookupTest, CanMoveConstruct)
{
    DensityFittingAmplitudeLookup lookup(DensityFittingAmplitudeMethod::Unity);
    DensityFittingAmplitudeLookup lookupCopied(std::move(lookup));
    const auto                    lookupResult = lookupCopied(charges_, masses_, lookupIndices_);
    EXPECT_EQ(lookupResult[0], 1);
    EXPECT_EQ(lookupResult[1], 1);
}

} // namespace gmx
