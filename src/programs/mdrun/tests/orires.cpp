/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 * Test for MD with orientation restraints
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{

class OriresTestFixture : public MdrunTestFixture
{
protected:
    OriresTestFixture();
    ~OriresTestFixture() override;
};

OriresTestFixture::OriresTestFixture() {}

OriresTestFixture::~OriresTestFixture() {}

//! Test fixture for mdrun with orires
typedef gmx::test::OriresTestFixture OriresTest;

/* Check whether the orires function works. */
TEST_F(OriresTest, OriresCanRun)
{
    runner_.useTopGroAndNdxFromDatabase("orires_1lvz");
    const std::string mdpContents = R"(
        dt            = 0.002
        nsteps        = 10
        tcoupl        = V-rescale
        tc-grps       = System
        tau-t         = 0.5
        ref-t         = 300
        constraints   = h-bonds
        cutoff-scheme = Verlet
        orire         = Yes
        orire-fitgrp  = backbone
    )";
    runner_.useStringAsMdpFile(mdpContents);

    EXPECT_EQ(0, runner_.callGrompp());

    ::gmx::test::CommandLine oriresCaller;

    // Do an mdrun with ORIRES enabled
    ASSERT_EQ(0, runner_.callMdrun(oriresCaller));
}

} // namespace test
} // namespace gmx
