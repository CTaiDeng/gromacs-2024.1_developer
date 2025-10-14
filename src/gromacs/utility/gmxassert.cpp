/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2011- The GROMACS Authors
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
 * Implements assertion handlers.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/gmxassert.h"

#include <cstdio>

#include "gromacs/utility/fatalerror.h"

#include "errorformat.h"

namespace gmx
{

/*! \cond internal */
namespace internal
{

void assertHandler(const char* condition, const char* msg, const char* func, const char* file, int line)
{
    printFatalErrorHeader(stderr, "Assertion failed", func, file, line);
    std::fprintf(stderr, "Condition: %s\n", condition);
    printFatalErrorMessageLine(stderr, msg, 0);
    printFatalErrorFooter(stderr);
    gmx_exit_on_fatal_error(ExitType_Abort, 1);
}

} // namespace internal
//! \endcond

} // namespace gmx
