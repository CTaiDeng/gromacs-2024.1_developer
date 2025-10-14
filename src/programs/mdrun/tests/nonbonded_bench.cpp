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
 * This implements basic Nonbonded bench tests.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "programs/mdrun/nonbonded_bench.h"

#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textreader.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{
namespace
{

TEST(NonbondedBenchTest, BasicEndToEndTest)
{
    const char* const command[] = { "nonbonded-benchmark" };
    CommandLine       cmdline(command);
    cmdline.addOption("-iter", 1);
    EXPECT_EQ(0,
              gmx::test::CommandLineTestHelper::runModuleFactory(
                      &gmx::NonbondedBenchmarkInfo::create, &cmdline));
}

} // namespace
} // namespace test
} // namespace gmx
