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
 * Declares the nonbonded benchmarking tool.
 *
 * \author Berk Hess <hess@kth.se>
 */

#ifndef GMX_PROGRAMS_MDRUN_NOBONDED_BENCH_MAIN_H
#define GMX_PROGRAMS_MDRUN_NOBONDED_BENCH_MAIN_H

#include "gromacs/commandline/cmdlineoptionsmodule.h"

namespace gmx
{

//! Declares gmx nonbonded-bench.
class NonbondedBenchmarkInfo
{
public:
    //! Name of the module.
    static const char name[];
    //! Short module description.
    static const char shortDescription[];
    //! Build the actual gmx module to use.
    static ICommandLineOptionsModulePointer create();
};

} // namespace gmx

#endif
