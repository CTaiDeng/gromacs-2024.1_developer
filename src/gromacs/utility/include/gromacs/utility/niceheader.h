/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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

/*! \libinternal \file
 * \brief
 * Declares function for printing a nice header for text output files.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_NICEHEADER_H
#define GMX_UTILITY_NICEHEADER_H

namespace gmx
{

class TextWriter;

/*! \brief
 * Prints creation time stamp and user information into a string as comments, and returns it.
 *
 * \param[out] writer         Where to print the information.
 * \param[in]  fn             Name of the file being written; if nullptr, described as "unknown".
 * \param[in]  commentChar    Character to use as the starting delimiter for comments.
 * \throws     std::bad_alloc if out of memory. */
void niceHeader(TextWriter* writer, const char* fn, char commentChar);

} // namespace gmx

#endif
