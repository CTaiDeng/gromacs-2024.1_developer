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
 * This implements tests on mdrun help writing.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "gromacs/commandline/cmdlinehelpcontext.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/cmdlinetest.h"
#include "testutils/refdata.h"

#include "programs/mdrun/mdrun_main.h"

namespace gmx
{
namespace test
{
namespace
{

TEST(MdrunTest, WritesHelp)
{
    // Make a stream to which we want gmx mdrun -h to write the help.
    StringOutputStream outputStream;
    TextWriter         writer(&outputStream);

    // Use that stream to set up a global help context. Legacy tools
    // like mdrun call parse_common_args, which recognizes the
    // existence of a global help context. That context triggers the
    // writing of help and a fast exit of the tool.
    HelpLinks*                   links = nullptr;
    CommandLineHelpContext       context(&writer, eHelpOutputFormat_Console, links, "dummy");
    GlobalCommandLineHelpContext global(context);

    // Call mdrun to get the help printed to the stream
    CommandLine caller;
    caller.append("mdrun");
    caller.append("-h");
    gmx_mdrun(caller.argc(), caller.argv());

    // Check whether the stream matches the reference copy.
    TestReferenceData    refData;
    TestReferenceChecker checker(refData.rootChecker());
    checker.checkString(outputStream.toString(), "Help string");
};

} // namespace
} // namespace test
} // namespace gmx
