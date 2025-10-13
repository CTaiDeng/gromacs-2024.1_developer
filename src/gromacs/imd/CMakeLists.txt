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

add_library(imd INTERFACE)
file(GLOB IMD_SOURCES *.cpp)
set(LIBGROMACS_SOURCES ${LIBGROMACS_SOURCES} ${IMD_SOURCES} PARENT_SCOPE)

# Source files have the following private module dependencies.
target_link_libraries(imd PRIVATE
#                      gmxlib
#                      math
#                      mdtypes
                      )

# Public interface for modules, including dependencies and interfaces
#target_include_directories(imd PUBLIC
target_include_directories(imd INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(imd PUBLIC
target_link_libraries(imd INTERFACE
                      legacy_api
                      )

# TODO: when imd is an OBJECT target
#target_link_libraries(imd PUBLIC legacy_api)
#target_link_libraries(imd PRIVATE common)

# Source files have the following private module dependencies.
#target_link_libraries(imd PRIVATE tng_io)
# TODO: Explicitly link specific modules.
#target_link_libraries(imd PRIVATE legacy_modules)
