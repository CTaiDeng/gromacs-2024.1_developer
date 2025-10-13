/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * Helper functions to have identical behavior of setenv and unsetenv
 * on Unix and Windows systems.
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \inlibraryapi
 * \ingroup module_testutils
 */

#include "config.h"

#include <cstdlib>

#ifndef GMX_TESTUTILS_SETENV_H
#    define GMX_TESTUTILS_SETENV_H

namespace gmx
{
namespace test
{
//! Workaround to make setenv work on Windows
inline int gmxSetenv(const char* name, const char* value, int overwrite)
{
#    if GMX_NATIVE_WINDOWS
    if (!overwrite)
    {
        size_t size  = 0;
        int    error = getenv_s(&size, nullptr, 0, name);
        if (error != 0 || size != 0)
        {
            return error;
        }
    }
    return _putenv_s(name, value);
#    else
    return setenv(name, value, overwrite);
#    endif
}

//! Workaround to make unsetenv work on Windows
inline int gmxUnsetenv(const char* name)
{
#    if GMX_NATIVE_WINDOWS
    return _putenv_s(name, "");
#    else
    return unsetenv(name);
#    endif
}
} // namespace test
} // namespace gmx

#endif // GMX_TESTUTILS_SETENV_H
