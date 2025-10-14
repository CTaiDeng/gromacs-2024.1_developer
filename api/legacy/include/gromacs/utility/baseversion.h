/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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
 * Declares functions to get basic version information.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inpublicapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_BASEVERSION_H
#define GMX_UTILITY_BASEVERSION_H

/*! \brief
 * Version string, containing the version, date, and abbreviated hash.
 *
 * This can be a plain version if git version info was disabled during the
 * build.
 * The returned string used to start with a literal word `VERSION` before
 * \Gromacs 2016, but no longer does.
 *
 * \ingroup module_utility
 */
const char* gmx_version();
/*! \brief
 * Full git hash of the latest commit.
 *
 * If git version info was disabled during the build, returns an empty string.
 *
 * \ingroup module_utility
 */
const char* gmx_version_git_full_hash();
/*! \brief
 * Full git hash of the latest commit in a central \Gromacs repository.
 *
 * If git version info was disabled during the build, returns an empty string.
 * Also, if the latest commit was from a central repository, the return value
 * is an empty string.
 *
 * \ingroup module_utility
 */
const char* gmx_version_git_central_base_hash();

/*! \brief
 * Defined if ``libgromacs`` has been compiled in double precision.
 *
 * Allows detecting the compiled precision of the library through checking the
 * presence of the symbol, e.g., from autoconf or other types of build systems.
 *
 * \ingroup module_utility
 */
void gmx_is_double_precision();
/*! \brief
 * Defined if ``libgromacs`` has been compiled in single/mixed precision.
 *
 * Allows detecting the compiled precision of the library through checking the
 * presence of the symbol, e.g., from autoconf or other types of build systems.
 *
 * \ingroup module_utility
 */

void gmx_is_single_precision();

/*! \brief Return a string describing what kind of GPU suport was configured in the build.
 *
 * Currently returns correctly for CUDA, OpenCL and SYCL.
 * Needs to be updated when adding new acceleration options.
 */
const char* getGpuImplementationString();

/*! \brief
 * DOI string, or empty when not a release build.
 */
const char* gmxDOI();

#endif
