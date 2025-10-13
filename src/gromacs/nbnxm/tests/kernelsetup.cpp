/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 * Tests for Setup of kernels.
 *
 * \author Joe Jordan <ejjordan@kth.se>
 * \ingroup module_nbnxm
 */
#include "gmxpre.h"

#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/nbnxm/kernel_common.h"
#include "gromacs/nbnxm/nbnxm.h"

#include "testutils/testasserts.h"

namespace gmx
{

namespace test
{

namespace
{

TEST(KernelSetupTest, getCoulombKernelTypeRF)
{
    EXPECT_EQ(getCoulombKernelType(Nbnxm::EwaldExclusionType::NotSet, CoulombInteractionType::RF, false),
              CoulombKernelType::ReactionField);
}

TEST(KernelSetupTest, getCoulombKernelTypeCut)
{
    EXPECT_EQ(getCoulombKernelType(Nbnxm::EwaldExclusionType::NotSet, CoulombInteractionType::Cut, false),
              CoulombKernelType::ReactionField);
}

TEST(KernelSetupTest, getCoulombKernelTypeTable)
{
    EXPECT_EQ(getCoulombKernelType(Nbnxm::EwaldExclusionType::Table, CoulombInteractionType::Count, true),
              CoulombKernelType::Table);
}

TEST(KernelSetupTest, getCoulombKernelTypeTableTwin)
{
    EXPECT_EQ(getCoulombKernelType(Nbnxm::EwaldExclusionType::Table, CoulombInteractionType::Count, false),
              CoulombKernelType::TableTwin);
}

TEST(KernelSetupTest, getCoulombKernelTypeEwald)
{
    EXPECT_EQ(getCoulombKernelType(Nbnxm::EwaldExclusionType::NotSet, CoulombInteractionType::Count, true),
              CoulombKernelType::Ewald);
}

TEST(KernelSetupTest, getCoulombKernelTypeEwaldTwin)
{
    EXPECT_EQ(getCoulombKernelType(Nbnxm::EwaldExclusionType::NotSet, CoulombInteractionType::Count, false),
              CoulombKernelType::EwaldTwin);
}

TEST(KernelSetupTest, getVdwKernelTypeLjCutCombGeomNone)
{
    EXPECT_EQ(getVdwKernelType(Nbnxm::KernelType::NotSet,
                               LJCombinationRule::Geometric,
                               VanDerWaalsType::Cut,
                               InteractionModifiers::None,
                               LongRangeVdW::Count),
              vdwktLJCUT_COMBGEOM);
}

TEST(KernelSetupTest, getVdwKernelTypeLjCutCombGeomPotShift)
{
    EXPECT_EQ(getVdwKernelType(Nbnxm::KernelType::NotSet,
                               LJCombinationRule::Geometric,
                               VanDerWaalsType::Cut,
                               InteractionModifiers::PotShift,
                               LongRangeVdW::Count),
              vdwktLJCUT_COMBGEOM);
}

TEST(KernelSetupTest, getVdwKernelTypeLjCutCombLBNone)
{
    EXPECT_EQ(getVdwKernelType(Nbnxm::KernelType::NotSet,
                               LJCombinationRule::LorentzBerthelot,
                               VanDerWaalsType::Cut,
                               InteractionModifiers::None,
                               LongRangeVdW::Count),
              vdwktLJCUT_COMBLB);
}

TEST(KernelSetupTest, getVdwKernelTypeLjCutCombLBPotShift)
{
    EXPECT_EQ(getVdwKernelType(Nbnxm::KernelType::NotSet,
                               LJCombinationRule::LorentzBerthelot,
                               VanDerWaalsType::Cut,
                               InteractionModifiers::PotShift,
                               LongRangeVdW::Count),
              vdwktLJCUT_COMBLB);
}

TEST(KernelSetupTest, getVdwKernelTypeLjCutCombNoneNone)
{
    EXPECT_EQ(getVdwKernelType(Nbnxm::KernelType::NotSet,
                               LJCombinationRule::None,
                               VanDerWaalsType::Cut,
                               InteractionModifiers::None,
                               LongRangeVdW::Count),
              vdwktLJCUT_COMBNONE);
}

TEST(KernelSetupTest, getVdwKernelTypeLjCutCombNonePotShift)
{
    EXPECT_EQ(getVdwKernelType(Nbnxm::KernelType::NotSet,
                               LJCombinationRule::None,
                               VanDerWaalsType::Cut,
                               InteractionModifiers::PotShift,
                               LongRangeVdW::Count),
              vdwktLJCUT_COMBNONE);
}

TEST(KernelSetupTest, getVdwKernelTypeLjCutThrows)
{
    EXPECT_ANY_THROW(getVdwKernelType(Nbnxm::KernelType::NotSet,
                                      LJCombinationRule::Count,
                                      VanDerWaalsType::Cut,
                                      InteractionModifiers::PotShift,
                                      LongRangeVdW::Count));
}

TEST(KernelSetupTest, getVdwKernelTypeLjCutForceSwitch)
{
    EXPECT_EQ(getVdwKernelType(Nbnxm::KernelType::NotSet,
                               LJCombinationRule::None,
                               VanDerWaalsType::Cut,
                               InteractionModifiers::ForceSwitch,
                               LongRangeVdW::Count),
              vdwktLJFORCESWITCH);
}

TEST(KernelSetupTest, getVdwKernelTypePmeGeom)
{
    EXPECT_EQ(getVdwKernelType(Nbnxm::KernelType::Cpu4x4_PlainC,
                               LJCombinationRule::None,
                               VanDerWaalsType::Pme,
                               InteractionModifiers::Count,
                               LongRangeVdW::Geom),
              vdwktLJEWALDCOMBGEOM);
}

TEST(KernelSetupTest, getVdwKernelTypePmeNone)
{
    EXPECT_EQ(getVdwKernelType(Nbnxm::KernelType::Cpu4x4_PlainC,
                               LJCombinationRule::None,
                               VanDerWaalsType::Pme,
                               InteractionModifiers::Count,
                               LongRangeVdW::Count),
              vdwktLJEWALDCOMBLB);
}

TEST(KernelSetupTest, getVdwKernelTypeLjCutPotSwitch)
{
    EXPECT_EQ(getVdwKernelType(Nbnxm::KernelType::NotSet,
                               LJCombinationRule::None,
                               VanDerWaalsType::Cut,
                               InteractionModifiers::PotSwitch,
                               LongRangeVdW::Count),
              vdwktLJPOTSWITCH);
}

TEST(KernelSetupTest, getVdwKernelTypeAllCountThrows)
{
    // Count cannot be used for VanDerWaalsType or InteractionModifiers because of calls to
    // enumValueToString(), which require a valid choice to have been made.
    EXPECT_ANY_THROW(getVdwKernelType(Nbnxm::KernelType::NotSet,
                                      LJCombinationRule::Count,
                                      VanDerWaalsType::Cut,
                                      InteractionModifiers::None,
                                      LongRangeVdW::Count));
}

} // namespace
} // namespace test
} // namespace gmx
