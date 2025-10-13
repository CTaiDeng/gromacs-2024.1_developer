# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2014- The GROMACS Authors
# Copyright (C) 2025- GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ---
#
# This file is part of a modified version of the GROMACS molecular simulation package.
# For details on the original project, consult https://www.gromacs.org.
#
# To help fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

if (NOT DEFINED GMX_EXECUTABLE)
    message(FATAL_ERROR "Required input parameter not set")
endif()

file(MAKE_DIRECTORY completion)
execute_process(
    COMMAND ${GMX_EXECUTABLE} -quiet help -export completion
    WORKING_DIRECTORY completion
    RESULT_VARIABLE exitcode)
if (exitcode)
    # Ensure that no partial output is left behind.
    file(REMOVE_RECURSE completion)
    if (ERRORS_ARE_FATAL)
        message(FATAL_ERROR
            "Failed to generate shell completions. "
            "Set GMX_BUILD_HELP=OFF if you want to skip the completions.\n"
            "Error/exit code: ${exitcode}")
    else()
        message(
            "Failed to generate shell completions, will build GROMACS without. "
            "Set GMX_BUILD_HELP=OFF if you want to skip this notification and "
            "warnings during installation.")
    endif()
endif()
