/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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
 * Declares gmx::ShellCompletionWriter.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_commandline
 */
#ifndef GMX_COMMANDLINE_SHELLCOMPLETIONS_H
#define GMX_COMMANDLINE_SHELLCOMPLETIONS_H

#include <memory>
#include <string>
#include <vector>

namespace gmx
{

class CommandLineHelpContext;
class Options;
class TextWriter;

//! \cond internal
//! \addtogroup module_commandline
//! \{

//! Output format for ShellCompletionWriter.
enum ShellCompletionFormat
{
    eShellCompletionFormat_Bash //!< Shell completions for bash.
};

//! \}
//! \endcond

class ShellCompletionWriter
{
public:
    typedef std::vector<std::string> ModuleNameList;

    ShellCompletionWriter(const std::string& binaryName, ShellCompletionFormat format);
    ~ShellCompletionWriter();

    TextWriter& outputWriter();

    void startCompletions();
    void writeModuleCompletions(const char* moduleName, const Options& options);
    void writeWrapperCompletions(const ModuleNameList& modules, const Options& options);
    void finishCompletions();

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
