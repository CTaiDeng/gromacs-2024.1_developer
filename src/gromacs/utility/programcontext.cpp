/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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
 * Implements gmx::IProgramContext and related methods.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/programcontext.h"

#include <cstddef>

namespace gmx
{

namespace
{

//! \addtogroup module_utility
//! \{

/*! \brief
 * Default implementation of IProgramContext.
 *
 * This implementation is used if nothing has been set with
 * setProgramContext().
 *
 * Since it is constructed using a global initializer, it should not throw.
 */
class DefaultProgramContext : public IProgramContext
{
public:
    DefaultProgramContext() {}

    const char*            programName() const override { return "GROMACS"; }
    const char*            displayName() const override { return "GROMACS"; }
    std::filesystem::path  fullBinaryPath() const override { return ""; }
    InstallationPrefixInfo installationPrefix() const override
    {
        return InstallationPrefixInfo("", false);
    }
    const char* commandLine() const override { return ""; }
};

//! Global program info; stores the object set with setProgramContext().
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const IProgramContext* g_programContext;
//! Default program context if nothing is set.
const DefaultProgramContext g_defaultContext;

//! \}

} // namespace

const IProgramContext& getProgramContext()
{
    if (g_programContext != nullptr)
    {
        return *g_programContext;
    }
    return g_defaultContext;
}

void setProgramContext(const IProgramContext* programContext)
{
    g_programContext = programContext;
}

} // namespace gmx
