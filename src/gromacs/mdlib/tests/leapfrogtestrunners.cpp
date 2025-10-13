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
 * \brief Runner for CPU-based implementation of the integrator.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "leapfrogtestrunners.h"

#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/utility/arrayref.h"

#include "testutils/testasserts.h"

#include "leapfrogtestdata.h"


namespace gmx
{
namespace test
{

void LeapFrogHostTestRunner::integrate(LeapFrogTestData* testData, int numSteps)
{
    testData->state_.x.resizeWithPadding(testData->numAtoms_);
    testData->state_.v.resizeWithPadding(testData->numAtoms_);
    for (int i = 0; i < testData->numAtoms_; i++)
    {
        testData->state_.x[i] = testData->x_[i];
        testData->state_.v[i] = testData->v_[i];
    }

    gmx_omp_nthreads_set(ModuleMultiThread::Update, 1);

    for (int step = 0; step < numSteps; step++)
    {
        testData->update_->update_coords(testData->inputRecord_,
                                         step,
                                         testData->mdAtoms_.homenr,
                                         testData->mdAtoms_.havePartiallyFrozenAtoms,
                                         testData->mdAtoms_.ptype,
                                         testData->mdAtoms_.invmass,
                                         testData->mdAtoms_.invMassPerDim,
                                         &testData->state_,
                                         testData->f_,
                                         &testData->forceCalculationData_,
                                         &testData->kineticEnergyData_,
                                         testData->velocityScalingMatrix_,
                                         etrtNONE,
                                         nullptr,
                                         false);
        testData->update_->finish_update(testData->inputRecord_,
                                         testData->mdAtoms_.havePartiallyFrozenAtoms,
                                         testData->mdAtoms_.homenr,
                                         &testData->state_,
                                         nullptr,
                                         false);
    }
    const auto xp = makeArrayRef(*testData->update_->xp()).subArray(0, testData->numAtoms_);
    for (int i = 0; i < testData->numAtoms_; i++)
    {
        for (int d = 0; d < DIM; d++)
        {
            testData->x_[i][d]      = testData->state_.x[i][d];
            testData->v_[i][d]      = testData->state_.v[i][d];
            testData->xPrime_[i][d] = xp[i][d];
        }
    }
}

} // namespace test
} // namespace gmx
