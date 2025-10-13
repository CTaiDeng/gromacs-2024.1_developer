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
 * \brief Declares functionality used to test mdrun termination
 * functionality under different conditions
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#ifndef GMX_MDRUN_TESTS_TERMINATIONHELPER_H
#define GMX_MDRUN_TESTS_TERMINATIONHELPER_H

#include "testutils/cmdlinetest.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{

/*! \internal
 * \brief Help test mdrun termination behaviour
 *
 * This helper class provides re-usable code to ensure that some
 * termination behaviour of mdrun works. It runs a simulation that
 * stops after a very short -maxh time, writes a checkpoint, checks
 * that the checkpoint exists, and then restarts with it (probably
 * doing no MD steps in the restart).
 *
 * \todo This approach is not very elegant, but "stuff doesn't
 * segfault or give a fatal error" is a useful result. We can improve
 * it when we can mock out more do_md() functionality. Before that,
 * we'd probably prefer not to run this test case in per-patchset
 * verification, but this is the best we can do for now.
 *
 * \ingroup module_mdrun_integration_tests
 */
class TerminationHelper
{
public:
    //! Constructor
    TerminationHelper(CommandLine* mdrunCaller, SimulationRunner* runner);
    /*! \brief Do a short simulation, likely terminated by -maxh
     *
     * \param[in] expectedCptFileName The name of the checkpoint
     * file that mdrun will write (which has to be customizable,
     * if we are testing a multi-simulation). */
    void runFirstMdrun(const std::string& expectedCptFileName);
    //! Check that the restart works, but don't do any more MD steps.
    void runSecondMdrun();
    //! Check that the restart works without appending, but don't do any more MD steps.
    void runSecondMdrunWithNoAppend();

protected:
    //! Object to help call mdrun
    CommandLine* mdrunCaller_;
    //! Object to coordinate running a simulation
    SimulationRunner* runner_;
};

} // namespace test
} // namespace gmx

#endif
