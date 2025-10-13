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

add_library(pulling OBJECT)
target_sources(pulling PRIVATE
        output.cpp
        pull.cpp
        pull_rotation.cpp
        pullcoordexpressionparser.cpp
        pullutil.cpp
        transformationcoordinate.cpp)
if (BUILD_SHARED_LIBS)
    set_target_properties(pulling PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Set required compile flags
gmx_target_compile_options(pulling)
target_compile_definitions(pulling PRIVATE HAVE_CONFIG_H)

# Source files have the following dependencies on library infrastructure.
target_link_libraries(pulling PRIVATE
                      common
                      legacy_modules
        math
        utility
        muparser
        linearalgebra
        timing
        topology
        utility
)

# pulling needs muparser defines 
target_link_libraries(pulling INTERFACE muparser::muparser)

# Public interface for modules, including dependencies and interfaces
target_include_directories(pulling PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(pulling PUBLIC
target_link_libraries(pulling INTERFACE
                      legacy_api
                      )

if (GMX_OPENMP)
    target_link_libraries(pulling PRIVATE OpenMP::OpenMP_CXX)
endif()

# TODO: when pulling is an OBJECT target
#target_link_libraries(pulling PUBLIC legacy_api)
#target_link_libraries(pulling PRIVATE common)

# Source files have the following private module dependencies.
#target_link_libraries(pulling PRIVATE NOTHING)
# TODO: Explicitly link specific modules.
#target_link_libraries(pulling PRIVATE legacy_modules)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
