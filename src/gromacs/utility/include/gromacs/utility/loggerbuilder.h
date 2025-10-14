/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * \brief
 * Declares functionality for initializing logging.
 *
 * See \ref page_logging for an overview of the functionality.
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_LOGGERBUILDER_H
#define GMX_UTILITY_LOGGERBUILDER_H

#include <memory>
#include <string>

#include "gromacs/utility/logger.h"

namespace gmx
{

class TextOutputStream;

class LoggerFormatterBuilder;
class LoggerOwner;

/*! \libinternal \brief
 * Initializes loggers.
 *
 * This class provides methods for specifying logging targets for a logger and
 * building the logger after all targets have been specified.  Having this
 * separate from the logger allows using different internal data structures
 * during initialization and operation, and simplifies the responsibilities of
 * the involved classes.
 *
 * \ingroup module_utility
 */
class LoggerBuilder
{
public:
    LoggerBuilder();
    ~LoggerBuilder();

    /*! \brief
     * Adds a stream to which log output is written.
     *
     * All output at level \p level or above it is written to \p stream.
     * The caller is responsible of closing and freeing \p stream once the
     * logger is discarded.
     */
    void addTargetStream(MDLogger::LogLevel level, TextOutputStream* stream);
    /*! \brief
     * Adds a file to which log output is written.
     *
     * All output at level \p level or above it is written to \p fp.
     * The caller is responsible of closing \p fp once the logger is
     * discarded.
     */
    void addTargetFile(MDLogger::LogLevel level, FILE* fp);

    /*! \brief
     * Builds the logger with the targets set for this builder.
     *
     * After this function has been called, the builder can (and should) be
     * discarded.
     */
    LoggerOwner build();

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

/*! \libinternal \brief
 * Manages memory for a logger built with LoggerBuilder.
 *
 * This class is responsible of managing all memory allocated by LoggerBuilder
 * that is needed for operation of the actual logger.  Also the actual logger
 * instance is owned by this class.  This allows keeping the actual logger simple
 * and streamlined.
 *
 * This class supports move construction and assignment, which allows
 * initializing it on the stack and assigning a new instance if the targets
 * need to be changed.
 *
 * \ingroup module_utility
 */
class LoggerOwner
{
public:
    //! Move-constructs the owner.
    LoggerOwner(LoggerOwner&& other) noexcept;
    ~LoggerOwner();

    //! Move-assings the owner.
    LoggerOwner& operator=(LoggerOwner&& other) noexcept;

    //! Returns the logger for writing the logs.
    const MDLogger& logger() const { return *logger_; }

private:
    class Impl;

    LoggerOwner(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
    const MDLogger*       logger_;

    friend class LoggerBuilder;
};

} // namespace gmx

#endif
