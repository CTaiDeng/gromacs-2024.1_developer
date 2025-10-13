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

/*! \file
 * \brief
 * Declares gmx::CommandLineParser.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inpublicapi
 * \ingroup module_commandline
 */
#ifndef GMX_COMMANDLINE_CMDLINEPARSER_H
#define GMX_COMMANDLINE_CMDLINEPARSER_H

#include <memory>
#include <string>
#include <vector>

namespace gmx
{

class Options;

/*! \brief
 * Implements command-line parsing for Options objects.
 *
 * Typical usage (without error checking):
 * \code
   gmx::Options options("name", "description");
   // Fill up options

   gmx::CommandLineParser(&options).parse(&argc, argv);
   options.finish();
 * \endcode
 *
 * \inpublicapi
 * \ingroup module_commandline
 */
class CommandLineParser
{
public:
    /*! \brief
     * Creates a command-line parser that sets values for options.
     *
     * \param[in] options  Options object whose options should be set.
     * \throws  std::bad_alloc if out of memory.
     */
    explicit CommandLineParser(Options* options);
    ~CommandLineParser();

    /*! \brief
     * Makes the parser skip unknown options and keep them in \c argv.
     *
     * \param[in] bEnabled  Whether to skip and keep unknown options.
     * \returns   *this
     *
     * Setting this option to true has dual effect: unknown options are
     * silently skipped, and all recognized options are removed from
     * \c argc and \c argv in parse().  These effects should be easy to
     * separate into different flags if there is need for it.
     *
     * The default is false: unknown options result in exceptions and
     * \c argc and \c argv are not modified.
     *
     * Does not throw.
     */
    CommandLineParser& skipUnknown(bool bEnabled);

    /*! \brief
     * Makes the parser accept positional arguments
     *
     * \param[in] bEnabled  Whether to skip and keep positional arguments.
     * \returns   *this
     *
     * Arguments that are not options (ie. no leading hyphen), and
     * which come before all options are acceptable if this has
     * been enabled. If so, these arguments are left in \c argc
     * and \c argv in parse().
     *
     * The default is false: unknown leading arguments result in
     * exceptions and \c argc and \c argv are not modified.
     *
     * Does not throw.
     */
    CommandLineParser& allowPositionalArguments(bool bEnabled);

    /*! \brief
     * Parses the command line.
     *
     * \throws  std::bad_alloc if out of memory.
     * \throws  InvalidInputError if any errors were detected in the input.
     *
     * All command-line arguments are parsed, and an aggregate
     * exception with all the detected errors (including unknown
     * options, where applicable) is thrown in the end.
     *
     * If skipUnknown() was not called, or last called with a
     * false value, the input arguments are not modified. If
     * skipUnknown() was last called with a true value, only
     * unknown options will be retained in \c argc and \c argv.
     *
     * All positional arguments are retained in the argument list,
     * but such arguments must precede all options.
     *
     * \c argv[0] is never modified.
     *
     */
    void parse(int* argc, char* argv[]);

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
