/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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
 * Declares variables that hold generated version information.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_BASEVERSION_GEN_H
#define GMX_UTILITY_BASEVERSION_GEN_H

/*! \cond internal */
//! \addtogroup module_utility
//! \{

//! Version string, containing the version, date, and abbreviated hash.
extern const char gmx_ver_string[];
//! Full git hash of the latest commit.
extern const char gmx_full_git_hash[];
//! Full git hash of the latest commit in a central \Gromacs repository.
extern const char gmx_central_base_hash[];
/*! \brief
 *  DOI string for the \Gromacs source code populated by CMake.
 *
 *  The variable is populated with the generated DOI string
 *  by CMake when the build of a release version is
 *  requested. Allows identification and
 *  referencing of different \Gromacs releases.
 */
extern const char gmxSourceDoiString[];

//! \}
//! \endcond

#endif
