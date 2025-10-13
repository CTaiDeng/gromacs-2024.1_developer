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

/*! \libinternal \file
 * \brief
 * Declares gmx::test::StdioTestHelper.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inlibraryapi
 * \ingroup module_testutils
 */
#ifndef GMX_TESTUTILS_STDIOHELPER_H
#define GMX_TESTUTILS_STDIOHELPER_H

#include <memory>

#include "gromacs/utility/classhelpers.h"

namespace gmx
{
namespace test
{

class TestFileManager;

/*! \libinternal \brief
 * Helper class for tests where code reads directly from `stdin`.
 *
 * Any method in this class may throw std::bad_alloc if out of memory.
 *
 * \inlibraryapi
 * \ingroup module_testutils
 */
class StdioTestHelper
{
public:
    //! Creates a helper using the given file manager.
    explicit StdioTestHelper(TestFileManager* fileManager) : fileManager_(*fileManager) {}
    //! Destructor
    ~StdioTestHelper();

    /*! \brief Accepts a string as input, writes it to a temporary
     * file and then reopens stdin to read the contents of that
     * string.
     *
     * \throws FileIOError  when the freopen() fails
     */
    void redirectStringToStdin(const char* theString);

private:
    TestFileManager& fileManager_;
    bool             redirected = false;

    GMX_DISALLOW_COPY_AND_ASSIGN(StdioTestHelper);
};

} // namespace test
} // namespace gmx

#endif
