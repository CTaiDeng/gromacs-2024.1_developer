/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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
 * Tests utilities for interactive molecular dynamics (IMD) setups.
 *
 * \author Carsten Kutzner <ckutzne@gwdg.de>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "gromacs/utility/stringutil.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{

class ImdTestFixture : public MdrunTestFixture, public ::testing::WithParamInterface<const char*>
{
protected:
    ImdTestFixture();
    ~ImdTestFixture() override;
};


ImdTestFixture::ImdTestFixture() {}

ImdTestFixture::~ImdTestFixture() {}


//! Test fixture for mdrun with IMD settings
typedef gmx::test::ImdTestFixture ImdTest;

/* This test checks
 * - whether the IMD-group parameter from the .mdp file is understood,
 * - whether mdrun understands the IMD-related command line parameters
     -imdpull, -imdwait, -imdterm,
 * - whether or not GROMACS was compiled with IMD support, that mdrun finishes
     without error when IMD is enabled in the TPR.
 *
 * TODO In future, consider checking that mdrun does not start IMD
 * when it should/can not.
 */
TEST_P(ImdTest, ImdCanRun)
{
    runner_.useTopGroAndNdxFromDatabase("glycine_vacuo");
    const std::string mdpContents = R"(
        dt            = 0.002
        nsteps        = 2
        tcoupl        = v-rescale
        tc-grps       = System
        tau-t         = 0.5
        ref-t         = 300
        cutoff-scheme = Verlet
        IMD-group     = Heavy_Atoms
        integrator    = %s
    )";
    // Interpolate the integrator selection into the .mdp file
    runner_.useStringAsMdpFile(formatString(mdpContents.c_str(), GetParam()));

    EXPECT_EQ(0, runner_.callGrompp());

    ::gmx::test::CommandLine imdCaller;
    imdCaller.addOption("-imdport", 0); // automatically assign a free port
    imdCaller.append("-imdpull");
    imdCaller.append("-noimdwait"); // cannot use -imdwait: then mdrun would not return control ...
    imdCaller.append("-noimdterm");

    // Do an mdrun with IMD enabled
    ASSERT_EQ(0, runner_.callMdrun(imdCaller));
}

// Check a dynamical integrator and an energy minimizer. No need to
// cover the whole space.
INSTANTIATE_TEST_SUITE_P(WithIntegrator, ImdTest, ::testing::Values("md", "steep"));

} // namespace test
} // namespace gmx
