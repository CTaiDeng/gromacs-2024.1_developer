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

/*! \internal \file
 * \brief Tests to verify that a simulator that only does some actions
 * periodically with propagators with constraints produces the expected results.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "config.h"

#include "periodicactions.h"

namespace gmx
{
namespace test
{

using ::testing::Combine;
using ::testing::Values;
using ::testing::ValuesIn;

// TODO The time for OpenCL kernel compilation means these tests time
// out. Once that compilation is cached for the whole process, these
// tests can run in such configurations.
#if !GMX_GPU_OPENCL
INSTANTIATE_TEST_SUITE_P(PropagatorsWithConstraints,
                         PeriodicActionsTest,
                         Combine(ValuesIn(propagationParametersWithConstraints()),
                                 Values(outputParameters)));
#else
INSTANTIATE_TEST_SUITE_P(DISABLED_PropagatorsWithConstraints,
                         PeriodicActionsTest,
                         Combine(ValuesIn(propagationParametersWithConstraints()),
                                 Values(outputParameters)));
#endif

} // namespace test
} // namespace gmx
