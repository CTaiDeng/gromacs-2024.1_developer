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

#include "gmxpre.h"

#include "threadaffinitytest.h"

#include "config.h"

#include <memory>

#include <gmock/gmock.h>

#include "gromacs/hardware/hardwaretopology.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/smalloc.h"

namespace gmx
{
namespace test
{

MockThreadAffinityAccess::MockThreadAffinityAccess() : supported_(true)
{
    using ::testing::_;
    using ::testing::Return;
    ON_CALL(*this, setCurrentThreadAffinityToCore(_)).WillByDefault(Return(true));
}

MockThreadAffinityAccess::~MockThreadAffinityAccess() {}


ThreadAffinityTestHelper::ThreadAffinityTestHelper()
{
    cr_.nnodes = gmx_node_num();
    cr_.nodeid = gmx_node_rank();
    // Default communicator is needed for [SIM]MAIN(cr) to work
    // TODO: Should get cleaned up once thread affinity works with
    //       communicators rather than the full cr (part of #2395)
    cr_.sizeOfDefaultCommunicator = gmx_node_num();
    cr_.rankInDefaultCommunicator = gmx_node_rank();
    cr_.duty                      = DUTY_PP;
#if GMX_MPI
    cr_.mpi_comm_mysim = MPI_COMM_WORLD;
#endif
    hwOpt_.threadAffinity      = ThreadAffinity::Auto;
    hwOpt_.totNumThreadsIsAuto = false;
    physicalNodeId_            = 0;
}

ThreadAffinityTestHelper::~ThreadAffinityTestHelper() = default;

void ThreadAffinityTestHelper::setLogicalProcessorCount(int logicalProcessorCount)
{
    hwTop_ = std::make_unique<HardwareTopology>(logicalProcessorCount);
}

} // namespace test
} // namespace gmx
