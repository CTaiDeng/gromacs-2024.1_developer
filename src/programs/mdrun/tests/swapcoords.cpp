/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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
 * Tests utilities for "Computational Electrophysiology" setups.
 *
 * \author Carsten Kutzner <ckutzne@gwdg.de>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{

class SwapTestFixture : public MdrunTestFixture
{
protected:
    SwapTestFixture();
    ~SwapTestFixture() override;
};


SwapTestFixture::SwapTestFixture() {}

SwapTestFixture::~SwapTestFixture() {}


//! Test fixture for mdrun with "Computational Electrophysiology" settings,
// i.e. double membrane sandwich with ion/water exchange protocol
typedef gmx::test::SwapTestFixture CompelTest;

/* This test ensures that the compel protocol can be run, that all of
 * the swapcoords parameters from the .mdp file are understood, and that
 * the swap state variables can be written to and read from checkpoint. */
TEST_F(CompelTest, SwapCanRun)
{
    runner_.useTopGroAndNdxFromDatabase("OctaneSandwich");
    const std::string mdpContents = R"(
        dt                       = 0.005
        nsteps                   = 2
        tcoupl                   = V-rescale
        tc-grps                  = System
        tau-t                    = 0.5
        ref-t                    = 300
        constraints              = all-bonds
        cutoff-scheme            = Verlet
        swapcoords               = Z
        swap_frequency           = 1
        split_group0             = Ch0
        split_group1             = Ch1
        massw_split0             = yes
        massw_split1             = no
        solvent_group            = SOL
        cyl0_r                   = 1
        cyl0_up                  = 0.5
        cyl0_down                = 0.5
        cyl1_r                   = 1
        cyl1_up                  = 0.5
        cyl1_down                = 0.5
        coupl_steps              = 5
        iontypes                 = 2
        iontype0-name            = NA+
        iontype0-in-A            = 8
        iontype0-in-B            = 11
        iontype1-name            = CL-
        iontype1-in-A            = -1
        iontype1-in-B            = -1
        threshold                = 1
     )";

    runner_.useStringAsMdpFile(mdpContents);

    EXPECT_EQ(0, runner_.callGrompp());

    runner_.swapFileName_ = fileManager_.getTemporaryFilePath("swap.xvg").u8string();

    ::gmx::test::CommandLine swapCaller;
    swapCaller.addOption("-swap", runner_.swapFileName_);

    // Do an initial mdrun that writes a checkpoint file
    ::gmx::test::CommandLine firstCaller(swapCaller);
    ASSERT_EQ(0, runner_.callMdrun(firstCaller));
    // Continue mdrun from that checkpoint file
    ::gmx::test::CommandLine secondCaller(swapCaller);
    secondCaller.addOption("-cpi", runner_.cptOutputFileName_);
    runner_.nsteps_ = 2;
    ASSERT_EQ(0, runner_.callMdrun(secondCaller));
}


/*! \todo Add other tests for the compel module, e.g.
 *
 *  - a test that checks that actually ion/water swaps have been done, by
 *    calling gmxcheck on the swap output file and a reference file
 */


} // namespace test
} // namespace gmx
