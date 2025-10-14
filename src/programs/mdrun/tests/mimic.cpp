/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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
 * Tests for MiMiC forces computation
 *
 * \author Viacheslav Bolnykh <v.bolnykh@hpc-leap.eu>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include <gtest/gtest.h>

#include "gromacs/topology/ifunc.h"
#include "gromacs/trajectory/energyframe.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/cmdlinetest.h"
#include "testutils/refdata.h"
#include "testutils/simulationdatabase.h"

#include "energycomparison.h"
#include "energyreader.h"
#include "moduletest.h"

namespace gmx
{
namespace test
{

//! Test fixture for bonded interactions
class MimicTest : public gmx::test::MdrunTestFixture
{
public:
    //! Execute the trajectory writing test
    void setupGrompp(const char* index_file, const char* top_file, const char* gro_file)
    {
        runner_.topFileName_ = TestFileManager::getInputFilePath(top_file).u8string();
        runner_.groFileName_ = TestFileManager::getInputFilePath(gro_file).u8string();
        runner_.ndxFileName_ = TestFileManager::getInputFilePath(index_file).u8string();
        runner_.useStringAsMdpFile(
                "integrator                = mimic\n"
                "QMMM-grps                 = QMatoms");
    }
    //! Prepare an mdrun caller
    CommandLine setupRerun()
    {
        CommandLine rerunCaller;
        rerunCaller.append("mdrun");
        rerunCaller.addOption("-rerun", runner_.groFileName_);
        runner_.edrFileName_ = fileManager_.getTemporaryFilePath(".edr").u8string();
        return rerunCaller;
    }
    //! Check the output of mdrun
    void checkRerun()
    {
        EnergyTermsToCompare energyTermsToCompare{ {
                { interaction_function[F_EPOT].longname, relativeToleranceAsFloatingPoint(-20.1, 1e-4) },
        } };

        TestReferenceData refData;
        auto              checker = refData.rootChecker();
        checkEnergiesAgainstReferenceData(runner_.edrFileName_, energyTermsToCompare, &checker);
    }
};

// This test checks if the energies produced with one quantum molecule are reasonable
TEST_F(MimicTest, OneQuantumMol)
{
    setupGrompp("1quantum.ndx", "4water.top", "4water.gro");
    ASSERT_EQ(0, runner_.callGrompp());

    test::CommandLine rerunCaller = setupRerun();

    ASSERT_EQ(0, runner_.callMdrun(rerunCaller));
    if (gmx_node_rank() == 0)
    {
        checkRerun();
    }
}

// This test checks if the energies produced with all quantum molecules are reasonable (0)
TEST_F(MimicTest, AllQuantumMol)
{
    setupGrompp("allquantum.ndx", "4water.top", "4water.gro");
    ASSERT_EQ(0, runner_.callGrompp());

    test::CommandLine rerunCaller = setupRerun();
    ASSERT_EQ(0, runner_.callMdrun(rerunCaller));
    if (gmx_node_rank() == 0)
    {
        checkRerun();
    }
}

// This test checks if the energies produced with two quantum molecules are reasonable
// Needed to check the LJ intermolecular exclusions
TEST_F(MimicTest, TwoQuantumMol)
{
    setupGrompp("2quantum.ndx", "4water.top", "4water.gro");
    ASSERT_EQ(0, runner_.callGrompp());

    test::CommandLine rerunCaller = setupRerun();
    ASSERT_EQ(0, runner_.callMdrun(rerunCaller));
    if (gmx_node_rank() == 0)
    {
        checkRerun();
    }
}

// This test checks if the energies produced with QM/MM boundary cutting the bond are ok
TEST_F(MimicTest, BondCuts)
{
    setupGrompp("ala.ndx", "ala.top", "ala.gro");
    ASSERT_EQ(0, runner_.callGrompp());

    test::CommandLine rerunCaller = setupRerun();
    ASSERT_EQ(0, runner_.callMdrun(rerunCaller));
    if (gmx_node_rank() == 0)
    {
        checkRerun();
    }
}

} // namespace test

} // namespace gmx
