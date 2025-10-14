/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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
 * Implements classes in analysissettings.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/analysissettings.h"

#include "gromacs/commandline/cmdlineoptionsmodule.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/gmxassert.h"

#include "analysissettings_impl.h"

namespace gmx
{


/********************************************************************
 * TrajectoryAnalysisSettings
 */

TrajectoryAnalysisSettings::TrajectoryAnalysisSettings() : impl_(new Impl)
{
    impl_->frflags |= TRX_NEED_X;
}


TrajectoryAnalysisSettings::~TrajectoryAnalysisSettings() {}


void TrajectoryAnalysisSettings::setOptionsModuleSettings(ICommandLineOptionsModuleSettings* settings)
{
    impl_->optionsModuleSettings_ = settings;
}


TimeUnit TrajectoryAnalysisSettings::timeUnit() const
{
    return impl_->timeUnit;
}


const AnalysisDataPlotSettings& TrajectoryAnalysisSettings::plotSettings() const
{
    return impl_->plotSettings;
}


unsigned long TrajectoryAnalysisSettings::flags() const
{
    return impl_->flags;
}


bool TrajectoryAnalysisSettings::hasFlag(unsigned long flag) const
{
    return (impl_->flags & flag) != 0U;
}


bool TrajectoryAnalysisSettings::hasPBC() const
{
    return impl_->bPBC;
}


bool TrajectoryAnalysisSettings::hasRmPBC() const
{
    return impl_->bRmPBC;
}


int TrajectoryAnalysisSettings::frflags() const
{
    return impl_->frflags;
}


void TrajectoryAnalysisSettings::setFlags(unsigned long flags)
{
    impl_->flags = flags;
}


void TrajectoryAnalysisSettings::setFlag(unsigned long flag, bool bSet)
{
    if (bSet)
    {
        impl_->flags |= flag;
    }
    else
    {
        impl_->flags &= ~flag;
    }
}


void TrajectoryAnalysisSettings::setPBC(bool bPBC)
{
    impl_->bPBC = bPBC;
}


void TrajectoryAnalysisSettings::setRmPBC(bool bRmPBC)
{
    impl_->bRmPBC = bRmPBC;
}


void TrajectoryAnalysisSettings::setFrameFlags(int frflags)
{
    impl_->frflags = frflags;
}

void TrajectoryAnalysisSettings::setHelpText(const ArrayRef<const char* const>& help)
{
    GMX_RELEASE_ASSERT(impl_->optionsModuleSettings_ != nullptr,
                       "setHelpText() called in invalid context");
    impl_->optionsModuleSettings_->setHelpText(help);
}

} // namespace gmx
