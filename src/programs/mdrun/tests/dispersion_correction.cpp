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
 * Test for MD with dispersion correction.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{

class DispersionCorrectionTestFixture : public MdrunTestFixture
{
protected:
    DispersionCorrectionTestFixture();
    ~DispersionCorrectionTestFixture() override;
};

DispersionCorrectionTestFixture::DispersionCorrectionTestFixture() {}

DispersionCorrectionTestFixture::~DispersionCorrectionTestFixture() {}

//! Test fixture for mdrun with dispersion correction
typedef gmx::test::DispersionCorrectionTestFixture DispersionCorrectionTest;

/* Check whether the dispersion correction function works. */
TEST_F(DispersionCorrectionTest, DispersionCorrectionCanRun)
{
    runner_.useTopGroAndNdxFromDatabase("alanine_vsite_vacuo");
    const std::string mdpContents = R"(
        dt            = 0.002
        nsteps        = 200
        tcoupl        = V-rescale
        tc-grps       = System
        tau-t         = 0.5
        ref-t         = 300
        constraints   = h-bonds
        cutoff-scheme = Verlet
        DispCorr      = AllEnerPres
    )";
    runner_.useStringAsMdpFile(mdpContents);

    EXPECT_EQ(0, runner_.callGrompp());

    ::gmx::test::CommandLine disperCorrCaller;

    // Do an mdrun with ORIRES enabled
    ASSERT_EQ(0, runner_.callMdrun(disperCorrCaller));
}

} // namespace test
} // namespace gmx
