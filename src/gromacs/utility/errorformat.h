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
 * Declares an internal helper function for formatting standard error messages.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_ERRORFORMAT_H
#define GMX_UTILITY_ERRORFORMAT_H

#include <cstdio>

namespace gmx
{

/*! \cond internal */
namespace internal
{

/*! \brief
 * Formats a common header for fatal error messages.
 *
 * Does not throw.
 *
 * \ingroup module_utility
 */
void printFatalErrorHeader(FILE* fp, const char* title, const char* func, const char* file, int line);
/*! \brief
 * Formats a line of fatal error message text.
 *
 * Does not throw.
 *
 * \ingroup module_utility
 */
void printFatalErrorMessageLine(FILE* fp, const char* text, int indent);
/*! \brief
 * Formats a common footer for fatal error messages.
 *
 * Does not throw.
 *
 * \ingroup module_utility
 */
void printFatalErrorFooter(FILE* fp);

} // namespace internal
//! \endcond

} // namespace gmx

#endif
