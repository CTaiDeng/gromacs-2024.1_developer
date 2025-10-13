/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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
 * \brief Defines the test runner for CPU version of SETTLE.
 *
 * Also adds stub for the GPU version to keep the compiler happy.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "settletestrunners.h"

#include "config.h"

#include <gtest/gtest.h>

#include "gromacs/mdlib/settle.h"

#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{

void SettleHostTestRunner::applySettle(SettleTestData*    testData,
                                       const t_pbc        pbc,
                                       const bool         updateVelocities,
                                       const bool         calcVirial,
                                       const std::string& testDescription)
{
    SettleData settled(testData->mtop_);

    settled.setConstraints(
            testData->idef_->il[F_SETTLE], testData->numAtoms_, testData->masses_, testData->inverseMasses_);

    bool errorOccured;
    int  numThreads  = 1;
    int  threadIndex = 0;
    csettle(settled,
            numThreads,
            threadIndex,
            &pbc,
            testData->x_.arrayRefWithPadding(),
            testData->xPrime_.arrayRefWithPadding(),
            testData->reciprocalTimeStep_,
            updateVelocities ? testData->v_.arrayRefWithPadding() : ArrayRefWithPadding<RVec>(),
            calcVirial,
            testData->virial_,
            &errorOccured);
    EXPECT_FALSE(errorOccured) << testDescription;
}

} // namespace test
} // namespace gmx
