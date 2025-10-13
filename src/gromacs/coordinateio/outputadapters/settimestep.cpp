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
 * Implements settimestep class.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */

#include "gmxpre.h"

#include "settimestep.h"

#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/exceptions.h"

namespace gmx
{

real SetTimeStep::calculateNewFrameTime(real currentInputFrameTime)
{
    real currentTime = 0.0;
    if (!haveProcessedFirstFrame_)
    {
        currentTime              = currentInputFrameTime;
        haveProcessedFirstFrame_ = true;
    }
    else
    {
        currentTime = previousFrameTime_ + timeStep_;
    }
    previousFrameTime_ = currentTime;

    return currentTime;
}

void SetTimeStep::processFrame(const int /* framenumber */, t_trxframe* input)
{
    input->time  = calculateNewFrameTime(input->time);
    input->bTime = true;
}

} // namespace gmx
