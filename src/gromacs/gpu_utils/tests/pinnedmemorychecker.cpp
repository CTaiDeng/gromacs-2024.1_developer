/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * Tests for GPU memory status checker isHostMemoryPinned() being correct.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 */
#include "gmxpre.h"

#include "config.h"

#if GMX_GPU_CUDA

#    include <vector>

#    include <gtest/gtest.h>

#    include "gromacs/gpu_utils/gpu_utils.h"
#    include "gromacs/gpu_utils/hostallocator.h"
#    include "gromacs/gpu_utils/pmalloc.h"
#    include "gromacs/utility/real.h"
#    include "gromacs/utility/smalloc.h"

#    include "testutils/test_hardware_environment.h"

namespace gmx
{

namespace test
{

namespace
{

//! Test fixture
using PinnedMemoryCheckerTest = ::testing::Test;

TEST_F(PinnedMemoryCheckerTest, DefaultContainerIsRecognized)
{
    /* Note that this tests can be executed even on hosts with no GPUs.
     * However, the checks for pending CUDA errors run cudaGetLastError(...),
     * which itself returns cudaErrorNoDevice in this case. This causes the
     * tests to crash. The conditionals in these tests should be removed
     * when a proper work-around for this problem is in place.
     */
    if (getTestHardwareEnvironment()->hasCompatibleDevices())
    {
        HostVector<real> dummy(3, 1.5);
        changePinningPolicy(&dummy, PinningPolicy::CannotBePinned);
        EXPECT_FALSE(isHostMemoryPinned(dummy.data()));
    }
}

TEST_F(PinnedMemoryCheckerTest, PinnedContainerIsRecognized)
{
    if (getTestHardwareEnvironment()->hasCompatibleDevices())
    {
        HostVector<real> dummy(3, 1.5);
        changePinningPolicy(&dummy, PinningPolicy::PinnedIfSupported);
        EXPECT_TRUE(isHostMemoryPinned(dummy.data()));
    }
}

TEST_F(PinnedMemoryCheckerTest, PinningChangesAreRecognized)
{
    if (getTestHardwareEnvironment()->hasCompatibleDevices())
    {
        HostVector<real> dummy(3, 1.5);
        changePinningPolicy(&dummy, PinningPolicy::PinnedIfSupported);
        EXPECT_TRUE(isHostMemoryPinned(dummy.data())) << "memory starts pinned";
        changePinningPolicy(&dummy, PinningPolicy::CannotBePinned);
        EXPECT_FALSE(isHostMemoryPinned(dummy.data())) << "memory is now unpinned";
        changePinningPolicy(&dummy, PinningPolicy::PinnedIfSupported);
        EXPECT_TRUE(isHostMemoryPinned(dummy.data())) << "memory is pinned again";
    }
}

TEST_F(PinnedMemoryCheckerTest, DefaultCBufferIsRecognized)
{
    if (getTestHardwareEnvironment()->hasCompatibleDevices())
    {
        real* dummy;
        snew(dummy, 3);
        EXPECT_FALSE(isHostMemoryPinned(dummy));
        sfree(dummy);
    }
}

TEST_F(PinnedMemoryCheckerTest, PinnedCBufferIsRecognized)
{
    if (getTestHardwareEnvironment()->hasCompatibleDevices())
    {
        real* dummy = nullptr;
        pmalloc(reinterpret_cast<void**>(&dummy), 3 * sizeof(real));
        EXPECT_TRUE(isHostMemoryPinned(dummy));
        pfree(dummy);
    }
}

} // namespace
} // namespace test
} // namespace gmx

#endif // GMX_GPU_CUDA
