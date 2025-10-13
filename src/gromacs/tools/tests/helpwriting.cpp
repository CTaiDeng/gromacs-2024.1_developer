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
 * This implements tests on tool help writing. Based on mdrun test version.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 */
#include "gmxpre.h"

#include <memory>

#include "gromacs/commandline/cmdlinehelpcontext.h"
#include "gromacs/commandline/cmdlinemodule.h"
#include "gromacs/commandline/cmdlineoptionsmodule.h"
#include "gromacs/tools/convert_tpr.h"
#include "gromacs/tools/dump.h"
#include "gromacs/tools/report_methods.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/cmdlinetest.h"
#include "testutils/refdata.h"

namespace gmx
{
namespace test
{
namespace
{

class HelpwritingTest : public gmx::test::CommandLineTestBase
{
public:
    void runTest(gmx::ICommandLineModule* module) { testWriteHelp(module); }
};

TEST_F(HelpwritingTest, ConvertTprWritesHelp)
{
    const std::unique_ptr<gmx::ICommandLineModule> module(gmx::ICommandLineOptionsModule::createModule(
            "convert-tpr", "Dummy Info", ConvertTprInfo::create()));
    runTest(module.get());
};


TEST_F(HelpwritingTest, DumpWritesHelp)
{
    const std::unique_ptr<gmx::ICommandLineModule> module(
            gmx::ICommandLineOptionsModule::createModule("dump", "Dummy Info", DumpInfo::create()));
    runTest(module.get());
};

TEST_F(HelpwritingTest, ReportMethodsWritesHelp)
{
    const std::unique_ptr<gmx::ICommandLineModule> module(gmx::ICommandLineOptionsModule::createModule(
            "report-methods", "Dummy Info", ReportMethodsInfo::create()));
    runTest(module.get());
};

} // namespace
} // namespace test
} // namespace gmx
