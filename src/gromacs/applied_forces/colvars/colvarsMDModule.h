/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2023- The GROMACS Authors
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
 * Declares factory structure for Colvars MDModule class
 *
 * \author Hubert Santuz <hubert.santuz@gmail.com>
 * \ingroup module_applied_forces
 */
#ifndef COLVARS_MDMODULE_H
#define COLVARS_MDMODULE_H

#include <memory>
#include <string>

namespace gmx
{

class IMDModule;

/*! \internal
    \brief Information about the colvars module.
 *
 * Provides name and method to create a colvars module.
 */
struct ColvarsModuleInfo
{
    /*! \brief
     * Creates a module for applying forces according to a colvar bias.
     */
    static std::unique_ptr<IMDModule> create();
    //! The name of the module
    static const std::string name_;
};


} // namespace gmx

#endif
