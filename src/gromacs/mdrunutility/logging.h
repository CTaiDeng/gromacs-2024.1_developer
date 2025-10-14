/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/*! \libinternal \file
 *
 * \brief Declares the MD log file handling routines.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inlibraryapi
 * \ingroup module_mdrunutility
 */
#ifndef GMX_MDRUNUTILITY_LOGGING_H
#define GMX_MDRUNUTILITY_LOGGING_H

#include <cstdio>

#include <memory>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/unique_cptr.h"

struct t_fileio;

namespace gmx
{

/*! \brief Close the log file */
void closeLogFile(t_fileio* logfio);

//! Simple guard pointer See unique_cptr for details.
using LogFilePtr = std::unique_ptr<t_fileio, functor_wrapper<t_fileio, closeLogFile>>;

/*! \brief Open the log file for writing/appending.
 *
 * \throws FileIOError when the log file cannot be opened. */
LogFilePtr openLogFile(const char* lognm, bool appendFiles);

/*! \brief Prepare to use the open log file when appending.
 *
 * Does not throw.
 */
void prepareLogAppending(FILE* fplog);

} // namespace gmx

#endif
