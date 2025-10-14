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

/*! \internal \file
 * \brief
 * Implements classes from cmdlinemodule.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_commandline
 */
#include "gmxpre.h"

#include "cmdlinemodule.h"

#include "gromacs/commandline/cmdlinehelpcontext.h"

namespace gmx
{

class CommandLineModuleSettings::Impl
{
public:
    Impl() : defaultNiceLevel_(19) {}

    int defaultNiceLevel_;
};

CommandLineModuleSettings::CommandLineModuleSettings() : impl_(new Impl) {}

CommandLineModuleSettings::~CommandLineModuleSettings() {}

int CommandLineModuleSettings::defaultNiceLevel() const
{
    return impl_->defaultNiceLevel_;
}

void CommandLineModuleSettings::setDefaultNiceLevel(int niceLevel)
{
    impl_->defaultNiceLevel_ = niceLevel;
}

//! \cond libapi
void writeCommandLineHelpCMain(const CommandLineHelpContext& context,
                               const char*                   name,
                               int (*mainFunction)(int argc, char* argv[]))
{
    char* argv[2];
    int   argc = 1;
    // TODO: The constness should not be cast away.
    argv[0] = const_cast<char*>(name);
    argv[1] = nullptr;
    GlobalCommandLineHelpContext global(context);
    mainFunction(argc, argv);
}
//! \endcond

} // namespace gmx
