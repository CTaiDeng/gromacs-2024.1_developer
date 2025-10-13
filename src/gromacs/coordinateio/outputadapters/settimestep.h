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

/*! \file
 * \brief
 * Declares gmx::SetTimeStep
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inpublicapi
 * \ingroup module_coordinateio
 */
#ifndef GMX_FILEIO_SETTIMESTEP_H
#define GMX_FILEIO_SETTIMESTEP_H

#include "gromacs/coordinateio/ioutputadapter.h"
#include "gromacs/utility/real.h"

namespace gmx
{

/*!\brief
 * SetTimeStep class allows changing trajectory time information.
 *
 * This class allows the user to set custom time step information for the
 * current frame in a trajectory.
 *
 * \inpublicapi
 * \ingroup module_coordinateio
 *
 */
class SetTimeStep : public IOutputAdapter
{
public:
    /*! \brief
     * Construct SetTime object with choice for how to change time.
     *
     * Can be used to initialize SetTime from outside of trajectoryanalysis
     * with the user specified option to change frame time information or not.
     *
     * \param[in] timeStep User defined value for the time step.
     */
    explicit SetTimeStep(real timeStep) :
        timeStep_(timeStep), previousFrameTime_(0.0), haveProcessedFirstFrame_(false)
    {
    }
    /*! \brief
     *  Move constructor for SetTimeStep.
     */
    SetTimeStep(SetTimeStep&& old) noexcept = default;

    ~SetTimeStep() override {}

    void processFrame(int framenumber, t_trxframe* input) override;

    void checkAbilityDependencies(unsigned long /* abilities */) const override {}

private:
    /*! \brief
     * Calculates the time of the current coordinate frame based on user input.
     *
     * If the current frame is the first one, no changes to the time are made.
     * For subsequent frames, the new frame time is based on the user input
     * and the time of the previous frame.
     *
     * \param[in] currentInputFrameTime Input from processed coordinate frame.
     */
    real calculateNewFrameTime(real currentInputFrameTime);

    //! Time difference between frames.
    real timeStep_;
    //! Time of the previous frame.
    real previousFrameTime_;
    //! Has the first frame been processed?
    bool haveProcessedFirstFrame_;
};

//! Smart pointer to manage the object.
using SetTimeStepPointer = std::unique_ptr<SetTimeStep>;

} // namespace gmx

#endif
