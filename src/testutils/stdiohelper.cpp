/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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
 * Implements classes in stdiohelper.h.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include "testutils/stdiohelper.h"

#include <cerrno>
#include <cstdio>

#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{

/********************************************************************
 * StdioTestHelper
 */

StdioTestHelper::~StdioTestHelper()
{
    if (redirected)
    {
        fclose(stdin);
    }
}

void StdioTestHelper::redirectStringToStdin(const char* theString)
{
    const std::string fakeStdin = fileManager_.getTemporaryFilePath(".stdin").u8string();
    gmx::TextWriter::writeFileFromString(fakeStdin, theString);
    if (nullptr == std::freopen(fakeStdin.c_str(), "r", stdin))
    {
        GMX_THROW_WITH_ERRNO(FileIOError("Failed to redirect a string to stdin"), "freopen", errno);
    }
    redirected = true;
}

} // namespace test
} // namespace gmx
