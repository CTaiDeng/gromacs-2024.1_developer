# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2012- The GROMACS Authors
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

set(EXECUTABLE_EXTENSION "")
if (GMX_NATIVE_WINDOWS OR GMX_CYGWIN)
    set(EXECUTABLE_EXTENSION ".exe")
endif()
set(PATH_SEARCH_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/test-bin)
file(MAKE_DIRECTORY ${PATH_SEARCH_TEST_DIR}/bin)
file(WRITE ${PATH_SEARCH_TEST_DIR}/bin/test-exe${EXECUTABLE_EXTENSION}
     "Test executable for path searching")
if (UNIX)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            test-exe
            ${PATH_SEARCH_TEST_DIR}/bin/test-rel-link)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${PATH_SEARCH_TEST_DIR}/bin/test-exe
            ${PATH_SEARCH_TEST_DIR}/bin/test-abs-link)
endif()

gmx_add_unit_test(CommandLineUnitTests commandline-test
    CPP_SOURCE_FILES
        cmdlinehelpmodule.cpp
        cmdlinehelpwriter.cpp
        cmdlinemodulemanager.cpp
        cmdlinemodulemanagertest.cpp
        cmdlineparser.cpp
        cmdlineprogramcontext.cpp
        filenm.cpp
        pargs.cpp
        )
target_link_libraries(commandline-test PRIVATE onlinehelp-test-shared)
