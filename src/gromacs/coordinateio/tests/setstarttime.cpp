/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

/*!\internal
 * \file
 * \brief
 * Tests for gmx::SetStartTime
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */


#include "gmxpre.h"

#include "gromacs/coordinateio/outputadapters/setstarttime.h"

#include <memory>

#include "gromacs/coordinateio/tests/coordinate_test.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/trajectory/trajectoryframe.h"

namespace gmx
{

namespace test
{

/*!\brief
 * Test fixture to prepare a setatoms object to pass data through.
 */
class SetStartTimeTest : public gmx::test::CommandLineTestBase
{
public:
    SetStartTimeTest() { clear_trxframe(frame(), true); }
    /*! \brief
     * Get access to the method for changing frame time information.
     *
     * \param[in] startTime User supplied start time to test.
     */
    SetStartTime* setStartTime(real startTime)
    {
        if (!setStartTime_)
        {
            setStartTime_ = std::make_unique<SetStartTime>(startTime);
        }
        return setStartTime_.get();
    }
    //! Get access to trajectoryframe to mess with.
    t_trxframe* frame() { return &frame_; }

private:
    //! Object to use for tests
    SetStartTimePointer setStartTime_;
    //! Storage of trajectoryframe.
    t_trxframe frame_;
};

TEST_F(SetStartTimeTest, WorksWithNonZeroStart)
{
    frame()->bTime = false;
    frame()->time  = 5;
    // Set step to nonsense value to check that it is ignored.
    SetStartTime* method = setStartTime(42);
    EXPECT_NO_THROW(method->processFrame(0, frame()));
    EXPECT_TRUE(frame()->bTime);
    EXPECT_EQ(frame()->time, 42);
    // Set frame time again to simulate advancing to new time.
    frame()->time = 15;
    EXPECT_NO_THROW(method->processFrame(1, frame()));
    EXPECT_EQ(frame()->time, 52);
    // Expect to use next frame time to get correct time step again.
    frame()->time = 20;
    EXPECT_NO_THROW(method->processFrame(2, frame()));
    EXPECT_EQ(frame()->time, 57);
}

TEST_F(SetStartTimeTest, WorksWithZeroStart)
{
    frame()->time        = 42;
    SetStartTime* method = setStartTime(0);
    EXPECT_NO_THROW(method->processFrame(0, frame()));
    EXPECT_EQ(frame()->time, 0);
    // No matter what the next time in the frame is, ignore it.
    frame()->time = 65;
    EXPECT_NO_THROW(method->processFrame(1, frame()));
    EXPECT_EQ(frame()->time, 23);
    // And so on for more frames.
    frame()->time = 72;
    EXPECT_NO_THROW(method->processFrame(2, frame()));
    EXPECT_EQ(frame()->time, 30);
}

} // namespace test

} // namespace gmx
