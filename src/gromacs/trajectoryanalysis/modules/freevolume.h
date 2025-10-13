/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Declares trajectory analysis module for free volume calculations.
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_trajectoryanalysis
 */
#ifndef GMX_TRAJECTORYANALYSIS_MODULES_FREEVOLUME_H
#define GMX_TRAJECTORYANALYSIS_MODULES_FREEVOLUME_H

#include "gromacs/trajectoryanalysis/analysismodule.h"

namespace gmx
{

namespace analysismodules
{

class FreeVolumeInfo
{
public:
    static const char                      name[];
    static const char                      shortDescription[];
    static TrajectoryAnalysisModulePointer create();
};

} // namespace analysismodules

} // namespace gmx

#endif
