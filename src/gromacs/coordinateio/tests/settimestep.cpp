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

/*!\internal
 * \file
 * \brief
 * Tests for gmx::SetTimeStep
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */


#include "gmxpre.h"

#include "gromacs/coordinateio/outputadapters/settimestep.h"

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
class SetTimeStepTest : public gmx::test::CommandLineTestBase
{
public:
    SetTimeStepTest() { clear_trxframe(frame(), true); }
    /*! \brief
     * Get access to the method for changing frame time information.
     *
     * \param[in] timeStep User supplied time step to test.
     */
    SetTimeStep* setTimeStep(real timeStep)
    {
        if (!setTimeStep_)
        {
            setTimeStep_ = std::make_unique<SetTimeStep>(timeStep);
        }
        return setTimeStep_.get();
    }
    //! Get access to trajectoryframe to mess with.
    t_trxframe* frame() { return &frame_; }

private:
    //! Object to use for tests
    SetTimeStepPointer setTimeStep_;
    //! Storage of trajectoryframe.
    t_trxframe frame_;
};

TEST_F(SetTimeStepTest, SetTimeStepWorks)
{
    frame()->time = 23;
    // Set start time to nonsense to make sure it is ignored.
    SetTimeStep* method = setTimeStep(7);
    EXPECT_NO_THROW(method->processFrame(0, frame()));
    EXPECT_EQ(frame()->time, 23);
    // No matter what the next time in the frame is, ignore it.
    frame()->time = 42;
    EXPECT_NO_THROW(method->processFrame(1, frame()));
    EXPECT_EQ(frame()->time, 30);
    // And so on for more frames.
    frame()->time = 0;
    EXPECT_NO_THROW(method->processFrame(2, frame()));
    EXPECT_EQ(frame()->time, 37);
}

} // namespace test

} // namespace gmx
