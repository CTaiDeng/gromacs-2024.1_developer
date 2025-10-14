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

/*!\internal
 * \file
 * \brief
 * Tests for processing of user input.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */

#include "gmxpre.h"

#include "requirements.h"

namespace gmx
{

namespace test
{

TEST_F(FlagTest, CanSetSimpleFlag)
{
    std::string option = "atoms";
    std::string value  = "always";
    setModuleFlag(option, value, &options_, TestEnums::efTestString);
    OutputRequirements reqs = requirementsBuilder_.process();
    EXPECT_EQ(reqs.atoms, ChangeAtomsType::Always);
}

TEST_F(FlagTest, CanAddNewBox)
{
    std::string option = "box";
    std::string value  = "3 3 3";
    setModuleFlag(option, value, &options_, TestEnums::efTestFloat);
    OutputRequirements req = requirementsBuilder_.process();
    EXPECT_EQ(req.box, ChangeFrameInfoType::Always);
}

TEST_F(FlagTest, SetsImplicitPrecisionChange)
{
    std::string option = "precision";
    std::string value  = "5";
    setModuleFlag(option, value, &options_, TestEnums::efTestInt);
    OutputRequirements req = requirementsBuilder_.process();
    EXPECT_EQ(req.precision, ChangeFrameInfoType::Always);
}

TEST_F(FlagTest, SetsImplicitStartTimeChange)
{
    std::string option = "starttime";
    std::string value  = "20";
    setModuleFlag(option, value, &options_, TestEnums::efTestFloat);
    OutputRequirements req = requirementsBuilder_.process();
    EXPECT_EQ(req.frameTime, ChangeFrameTimeType::StartTime);
}

TEST_F(FlagTest, SetsImplicitTimeStepChange)
{
    std::string option = "timestep";
    std::string value  = "20";
    setModuleFlag(option, value, &options_, TestEnums::efTestFloat);
    OutputRequirements req = requirementsBuilder_.process();
    EXPECT_EQ(req.frameTime, ChangeFrameTimeType::TimeStep);
}

} // namespace test

} // namespace gmx
