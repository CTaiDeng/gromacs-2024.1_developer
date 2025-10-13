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
 * Tests for CUDA float3 type layout.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#include "gmxpre.h"

#include "config.h"

#if GMX_GPU_CUDA

#    include "gromacs/gpu_utils/gputraits.h"
#    include "gromacs/hardware/device_management.h"
#    include "gromacs/utility/arrayref.h"
#    include "gromacs/utility/exceptions.h"

#    include "testutils/test_hardware_environment.h"
#    include "testutils/testasserts.h"
#    include "testutils/testmatchers.h"

#    include "typecasts_runner.h"

namespace gmx
{

namespace test
{

//! Test data in RVec format
static const std::vector<RVec> rVecInput = { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };

TEST(GpuDataTypesCompatibilityTest, RVecAndFloat3Host)
{
    std::vector<RVec> rVecOutput(rVecInput.size());
    convertRVecToFloat3OnHost(rVecOutput, rVecInput);
    EXPECT_THAT(rVecInput, testing::Pointwise(RVecEq(ulpTolerance(0)), rVecOutput));
}

TEST(GpuDataTypesCompatibilityTest, RVecAndFloat3Device)
{
    for (const auto& testDevice : getTestHardwareEnvironment()->getTestDeviceList())
    {
        testDevice->activate();
        std::vector<RVec> rVecOutput(rVecInput.size());
        convertRVecToFloat3OnDevice(rVecOutput, rVecInput, testDevice.get());
        EXPECT_THAT(rVecInput, testing::Pointwise(RVecEq(ulpTolerance(0)), rVecOutput));
    }
}

} // namespace test
} // namespace gmx

#endif // GMX_GPU_CUDA
