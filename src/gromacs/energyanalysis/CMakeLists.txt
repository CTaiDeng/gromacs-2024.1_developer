# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2014- The GROMACS Authors
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

add_library(energyanalysis OBJECT)
target_sources(energyanalysis PRIVATE
               energyterm.cpp
               )

if (BUILD_SHARED_LIBS)
    set_target_properties(energyanalysis PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

gmx_target_compile_options(energyanalysis)
target_compile_definitions(energyanalysis PRIVATE HAVE_CONFIG_H)

# Source files have the following dependencies on library infrastructure.
target_link_libraries(energyanalysis PRIVATE common)

# Public interface for modules, including dependencies and interfaces
target_include_directories(energyanalysis PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(energyanalysis PUBLIC
                      fileio
                      legacy_api
                      utility
                      )

# Module dependencies
# energyanalysis interfaces convey transitive dependence on these modules.
target_link_libraries(energyanalysis PUBLIC
                      )
# Source files have the following private module dependencies.
#target_link_libraries(energyanalysis PRIVATE NOTHING)
# TODO: Explicitly link specific modules.
target_link_libraries(energyanalysis PRIVATE legacy_modules)

if(GMX_INSTALL_LEGACY_API)
    install(FILES
            DESTINATION include/gromacs/energyanalysis)
endif()

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
