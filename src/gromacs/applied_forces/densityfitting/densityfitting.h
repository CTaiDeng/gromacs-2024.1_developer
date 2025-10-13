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

#ifndef GMX_APPLIED_FORCES_DENSITYFITTING_H
#define GMX_APPLIED_FORCES_DENSITYFITTING_H

#include <memory>
#include <string>

namespace gmx
{

class IMDModule;

/*! \libinternal \brief Information about the density fitting module.
 *
 * Provides name and method to create a density fitting module.
 */
struct DensityFittingModuleInfo
{
    /*! \brief
     * Creates a module for applying forces to fit a given density.
     *
     * Fitting an all-atom structure into an experimental cryo-EM density map is a
     * typical application.
     */
    static std::unique_ptr<IMDModule> create();
    //! The name of the module
    static const std::string name_;
};

} // namespace gmx

#endif
