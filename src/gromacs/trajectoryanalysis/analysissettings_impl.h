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
 * Declares private implementation class for gmx::TrajectoryAnalysisSettings.
 *
 * \ingroup module_trajectoryanalysis
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 */
#ifndef GMX_TRAJECTORYANALYSIS_ANALYSISSETTINGS_IMPL_H
#define GMX_TRAJECTORYANALYSIS_ANALYSISSETTINGS_IMPL_H

#include <string>

#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/options/timeunitmanager.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"

namespace gmx
{

class ICommandLineOptionsModuleSettings;

/*! \internal \brief
 * Private implementation class for TrajectoryAnalysisSettings.
 *
 * \ingroup module_trajectoryanalysis
 */
class TrajectoryAnalysisSettings::Impl
{
public:
    //! Initializes the default values for the settings object.
    Impl() :
        timeUnit(TimeUnit::Default), flags(0), frflags(0), bRmPBC(true), bPBC(true), optionsModuleSettings_(nullptr)
    {
    }

    //! Global time unit setting for the analysis module.
    TimeUnit timeUnit;
    //! Global plotting settings for the analysis module.
    AnalysisDataPlotSettings plotSettings;
    //! Flags for the analysis module.
    unsigned long flags;
    //! Frame reading flags for the analysis module.
    int frflags;

    //! Whether to make molecules whole for each frame.
    bool bRmPBC;
    //! Whether to pass PBC information to the analysis module.
    bool bPBC;

    //! Lower-level settings object wrapped by these settings.
    ICommandLineOptionsModuleSettings* optionsModuleSettings_;
};

} // namespace gmx

#endif
