/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

/*!\internal
 * \file
 * \brief
 * Tests for outputmanager
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */

#include "gmxpre.h"

#include "outputadapters.h"

namespace gmx
{

namespace test
{

TEST_P(SetAtomsSupportedFiles, Works)
{
    prepareTest(GetParam());
}

TEST_P(SetAtomsUnSupportedFiles, Works)
{
    prepareTest(GetParam());
}

TEST_P(AnyOutputSupportedFiles, Works)
{
    prepareTest(GetParam());
}

TEST_F(OutputSelectorDeathTest, RejectsBadSelection)
{
    prepareTest();
}

TEST_P(SetVelocitySupportedFiles, Works)
{
    prepareTest(GetParam());
}

TEST_P(SetVelocityUnSupportedFiles, Works)
{
    prepareTest(GetParam());
}

TEST_P(SetForceSupportedFiles, Works)
{
    prepareTest(GetParam());
}

TEST_P(SetForceUnSupportedFiles, Works)
{
    prepareTest(GetParam());
}

TEST_P(SetPrecisionSupportedFiles, Works)
{
    prepareTest(GetParam());
}

TEST_P(SetPrecisionUnSupportedFiles, Works)
{
    prepareTest(GetParam());
}

TEST_P(NoOptionalOutput, Works)
{
    prepareTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(ModuleSupported, SetAtomsSupportedFiles, ::testing::ValuesIn(setAtomsSupported));

INSTANTIATE_TEST_SUITE_P(ModuleUnSupported, SetAtomsUnSupportedFiles, ::testing::ValuesIn(setAtomsUnSupported));

INSTANTIATE_TEST_SUITE_P(ModuleSupported, AnyOutputSupportedFiles, ::testing::ValuesIn(anySupported));

INSTANTIATE_TEST_SUITE_P(ModuleSupported, SetVelocitySupportedFiles, ::testing::ValuesIn(setVelocitySupported));

INSTANTIATE_TEST_SUITE_P(ModuleUnSupported,
                         SetVelocityUnSupportedFiles,
                         ::testing::ValuesIn(setVelocityUnSupported));

INSTANTIATE_TEST_SUITE_P(ModuleSupported, SetForceSupportedFiles, ::testing::ValuesIn(setForceSupported));

INSTANTIATE_TEST_SUITE_P(ModuleUnSupported, SetForceUnSupportedFiles, ::testing::ValuesIn(setForceUnSupported));

INSTANTIATE_TEST_SUITE_P(ModuleSupported,
                         SetPrecisionSupportedFiles,
                         ::testing::ValuesIn(setPrecisionSupported));

INSTANTIATE_TEST_SUITE_P(ModuleUnSupported,
                         SetPrecisionUnSupportedFiles,
                         ::testing::ValuesIn(setPrecisionUnSupported));

INSTANTIATE_TEST_SUITE_P(ModuleSupported, NoOptionalOutput, ::testing::ValuesIn(anySupported));
} // namespace test

} // namespace gmx
