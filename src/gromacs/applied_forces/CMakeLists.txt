# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2015- The GROMACS Authors
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

add_library(applied_forces INTERFACE)

# Source files have the following private module dependencies.
target_link_libraries(applied_forces PRIVATE
                      #                      gmxlib
                      #                      math
                      #                      mdtypes
                      #                      tng_io
                      )

# Public interface for modules, including dependencies and interfaces
#target_include_directories(applied_forces PUBLIC
target_include_directories(applied_forces INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(applied_forces PUBLIC
target_link_libraries(applied_forces INTERFACE
                      legacy_api
                      )

# TODO: when fileio is an OBJECT target
#target_link_libraries(applied_forces PUBLIC legacy_api)
#target_link_libraries(applied_forces PRIVATE common)

# Source files have the following private module dependencies.
#target_link_libraries(applied_forces PRIVATE tng_io)
# TODO: Explicitly link specific modules.
#target_link_libraries(applied_forces PRIVATE legacy_modules)

gmx_add_libgromacs_sources(
    electricfield.cpp
    )

add_subdirectory(awh)
add_subdirectory(densityfitting)
add_subdirectory(qmmm)
add_subdirectory(colvars)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
