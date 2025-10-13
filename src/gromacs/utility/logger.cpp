/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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

#include "gmxpre.h"

#include "gromacs/utility/logger.h"

#include <cstdarg>

#include "gromacs/utility/stringutil.h"

namespace gmx
{

namespace
{

//! Helper method for reading logging targets from an array.
ILogTarget* getTarget(ILogTarget* targets[MDLogger::LogLevelCount], MDLogger::LogLevel level)
{
    return targets[static_cast<int>(level)];
}

} // namespace

ILogTarget::~ILogTarget() {}


LogEntryWriter& LogEntryWriter::appendTextFormatted(gmx_fmtstr const char* fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    entry_.text.append(formatStringV(fmt, ap));
    va_end(ap);
    return *this;
}

MDLogger::MDLogger() :
    warning(nullptr), error(nullptr), debug(nullptr), verboseDebug(nullptr), info(nullptr)
{
}

MDLogger::MDLogger(ILogTarget* targets[LogLevelCount]) :
    warning(getTarget(targets, LogLevel::Warning)),
    error(getTarget(targets, LogLevel::Error)),
    debug(getTarget(targets, LogLevel::Debug)),
    verboseDebug(getTarget(targets, LogLevel::VerboseDebug)),
    info(getTarget(targets, LogLevel::Info))
{
}

} // namespace gmx
