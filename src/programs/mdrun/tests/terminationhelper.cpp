/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * \brief Defines functionality used to test mdrun termination
 * functionality under different conditions
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "terminationhelper.h"

#include <gtest/gtest.h>

#include "gromacs/utility/path.h"

#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{

TerminationHelper::TerminationHelper(CommandLine* mdrunCaller, SimulationRunner* runner) :
    mdrunCaller_(mdrunCaller), runner_(runner)
{
    runner_->useTopGroAndNdxFromDatabase("spc2");
}

void TerminationHelper::runFirstMdrun(const std::string& expectedCptFileName)
{
    CommandLine firstPart(*mdrunCaller_);
    // Stop after 0.036 ms, which should be short enough that
    // numSteps isn't reached first.
    firstPart.addOption("-maxh", 1e-7);
    firstPart.addOption("-nstlist", 1);
    ASSERT_EQ(0, runner_->callMdrun(firstPart));
    EXPECT_EQ(true, File::exists(expectedCptFileName, File::returnFalseOnError))
            << expectedCptFileName << " was not found";
}

void TerminationHelper::runSecondMdrun()
{
    CommandLine secondPart(*mdrunCaller_);
    secondPart.addOption("-cpi", runner_->cptOutputFileName_);
    secondPart.addOption("-nsteps", 2);
    ASSERT_EQ(0, runner_->callMdrun(secondPart));
}

void TerminationHelper::runSecondMdrunWithNoAppend()
{
    CommandLine secondPart(*mdrunCaller_);
    secondPart.addOption("-cpi", runner_->cptOutputFileName_);
    secondPart.addOption("-nsteps", 2);
    secondPart.append("-noappend");
    ASSERT_EQ(0, runner_->callMdrun(secondPart));
}

} // namespace test
} // namespace gmx
