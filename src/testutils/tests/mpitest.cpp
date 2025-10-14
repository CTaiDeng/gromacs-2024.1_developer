/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * Tests for infrastructure for running tests under MPI.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include "testutils/mpitest.h"

#include "config.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/gmxmpi.h"

namespace gmx
{
namespace test
{
namespace
{

class MpiSelfTest : public ::testing::Test
{
public:
    //! Whether each rank participated, relevant only on rank 0
    std::vector<int> reached_;
};

TEST_F(MpiSelfTest, Runs)
{
    GMX_MPI_TEST(RequireMinimumRankCount<2>);
    if (gmx_node_rank() == 0)
    {
        reached_.resize(getNumberOfTestMpiRanks(), 0);
    }
    // Needed for thread-MPI so that we resize the buffer before we
    // fill it on non-main ranks.
    MPI_Barrier(MPI_COMM_WORLD);
    int value = 1;
    MPI_Gather(&value, 1, MPI_INT, reached_.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (gmx_node_rank() == 0)
    {
        EXPECT_THAT(reached_, testing::Each(value));
    }
}

} // namespace
} // namespace test
} // namespace gmx
