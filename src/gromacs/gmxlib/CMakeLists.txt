# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2009- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
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

add_library(gmxlib INTERFACE)

add_subdirectory(nonbonded)

# The nonbonded directory contains subdirectories that are only
# conditionally built, so we cannot use a GLOB_RECURSE here.
file(GLOB GMXLIB_SOURCES *.cpp)

set(GMXLIB_SOURCES ${GMXLIB_SOURCES} ${NONBONDED_SOURCES} PARENT_SCOPE)

# TODO(#3288): Explicitly link specific modules for source dependencies when gmxlib is an OBJECT target.
# Source files have the following private module dependencies.
target_link_libraries(gmxlib PRIVATE
                      #                      common
                      #                      math
                      #                      mdtypes
                      #                      tng_io
                      )

# Public interface for modules, including dependencies and interfaces
#target_include_directories(gmxlib PUBLIC
target_include_directories(gmxlib INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(gmxlib PUBLIC
target_link_libraries(gmxlib INTERFACE
                      legacy_api
                      mdlib
                      pbcutil
                      utility
                      )

if(BUILD_TESTING)
    add_subdirectory(nonbonded/tests)
endif()
