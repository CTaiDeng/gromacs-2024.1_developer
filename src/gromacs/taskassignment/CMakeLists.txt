# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2017- The GROMACS Authors
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

add_library(taskassignment OBJECT)
target_sources(taskassignment PRIVATE
               decidegpuusage.cpp
               decidesimulationworkload.cpp
               findallgputasks.cpp
               reportgpuusage.cpp
               resourcedivision.cpp
               taskassignment.cpp
               usergpuids.cpp)
if (BUILD_SHARED_LIBS)
    set_target_properties(taskassignment PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

gmx_target_compile_options(taskassignment)
target_compile_definitions(taskassignment PRIVATE HAVE_CONFIG_H)

# Note that this is a higher-level module that should not need
# special compilation to suit e.g. GPU or MPI dependencies.
# If you find you want to do that, consider a preliminary
# refactoring.

# Source files have the following dependencies on library infrastructure.
target_link_libraries(taskassignment PRIVATE common)

# Public interface for modules, including dependencies and interfaces
target_include_directories(taskassignment PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(taskassignment PRIVATE
                      legacy_api
                      legacy_modules
                      math
                      pbcutil
                      pulling
                      timing
                      topology
                      utility
                      )

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
