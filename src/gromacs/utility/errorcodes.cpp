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
 * Implements functions in errorcodes.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "errorcodes.h"

namespace gmx
{

namespace
{

/*! \brief
 * Strings corresponding to gmx::ErrorCode values.
 *
 * This has to match the enum in errorcodes.h!
 *
 * \ingroup module_utility
 */
const char* const error_names[] = {
    "No error",
    "Out of memory",
    "File not found",
    "System I/O error",
    "Error in user input",
    "Inconsistency in user input",
    "Requested tolerance cannot be achieved",
    "Simulation instability detected",

    "Feature not implemented",
    "Invalid value (bug)",
    "Invalid call (bug)",
    "Internal error (bug)",
    "API error (bug)",
    "Range checking error (possible bug)",
    "Communication (parallel processing) problem",
    "Modular simulator error",

    "Unknown error",
};

} // namespace

const char* getErrorCodeString(int errorcode)
{
    if (errorcode < 0 || errorcode >= eeUnknownError)
    {
        errorcode = eeUnknownError;
    }
    return error_names[errorcode];
}

} // namespace gmx
