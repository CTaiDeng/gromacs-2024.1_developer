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

/*! \file
 * \brief
 * Declares gmx::SetStartTime
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inpublicapi
 * \ingroup module_coordinateio
 */
#ifndef GMX_FILEIO_SETSTARTTIME_H
#define GMX_FILEIO_SETSTARTTIME_H

#include "gromacs/coordinateio/ioutputadapter.h"
#include "gromacs/utility/real.h"

namespace gmx
{

/*!\brief
 * SetStartTime class allows changing trajectory time information.
 *
 * This class allows the user to set custom start time information for the
 * current frame in a trajectory.
 *
 * \inpublicapi
 * \ingroup module_coordinateio
 *
 */
class SetStartTime : public IOutputAdapter
{
public:
    /*! \brief
     * Construct object with choice for how to change initial time.
     *
     * \param[in] startTime User defined value for the initial time.
     */
    explicit SetStartTime(real startTime) :
        startTime_(startTime), haveProcessedFirstFrame_(false), differenceToInitialTime_(0)
    {
    }
    /*! \brief
     *  Move constructor for SetStartTime.
     */
    SetStartTime(SetStartTime&& old) noexcept = default;

    ~SetStartTime() override {}

    void processFrame(int /* framenumber */, t_trxframe* input) override;

    void checkAbilityDependencies(unsigned long /* abilities */) const override {}

private:
    /*! \brief
     * Set initial time from first processed frame.
     *
     * Calculates the time shift between the user set time and the time
     * in the coordinate frame being processed from the first processed coordinate
     * frame. This time shift is then used to calculate new frame times for each processed
     * coordinate frame.
     *
     * \param[in] initialTime Time value obtained from first frame.
     */
    void setInitialTime(real initialTime);

    /*! \brief
     * Stores the value of the initial time.
     *
     * In case users supply a new time step, the initial time of the
     * processed coordinate frame is stored here. In case the user also supplies
     * a new initial time, this variable is set to the user input instead.
     */
    real startTime_;
    //! Has the first frame been processed?
    bool haveProcessedFirstFrame_;
    /*! \brief
     * If the initial time is changed, we need to keep track of the initial
     * time difference to adjust the time of all following frames.
     */
    real differenceToInitialTime_;
};

//! Smart pointer to manage the object.
using SetStartTimePointer = std::unique_ptr<SetStartTime>;

} // namespace gmx

#endif
