# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2010- The GROMACS Authors
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

add_library(options OBJECT
    abstractoption.cpp
    abstractsection.cpp
    basicoptions.cpp
    behaviorcollection.cpp
    filenameoption.cpp
    filenameoptionmanager.cpp
    options.cpp
    optionsassigner.cpp
    optionsection.cpp
    optionsvisitor.cpp
    timeunitmanager.cpp
    treesupport.cpp
    )

gmx_target_compile_options(options)
target_compile_definitions(options PRIVATE HAVE_CONFIG_H)

# Source files have the following private module dependencies.
target_link_libraries(options PRIVATE
                legacy_api
                legacy_modules
                utility
                )

# Public interface for modules, including dependencies and interfaces
target_include_directories(options PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(options PRIVATE common)

if (BUILD_SHARED_LIBS)  
    set_target_properties(options PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
