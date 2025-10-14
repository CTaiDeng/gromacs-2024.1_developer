# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2019- The GROMACS Authors
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

file(GLOB MODULARSIMULATOR_SOURCES *.cpp)

add_library(modularsimulator OBJECT ${MODULARSIMULATOR_SOURCES})

if (GMX_OPENMP)
    # Explicitly set properties for modular simulator module to compile with openmp
    set_target_properties(modularsimulator PROPERTIES COMPILE_OPTIONS $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>)
endif ()

if (WIN32)
    gmx_target_warning_suppression(modularsimulator /wd4244 HAS_NO_MSVC_LOSSY_CONVERSION)
    gmx_target_warning_suppression(modularsimulator /wd4996 HAS_NO_MSVC_UNSAFE_FUNCTION)
endif ()

target_include_directories(modularsimulator SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/src/external)
gmx_target_compile_options(modularsimulator)
target_compile_definitions(modularsimulator PRIVATE HAVE_CONFIG_H)
# Should be possible to remove this when resolving #3290
target_include_directories(modularsimulator SYSTEM BEFORE PRIVATE ${PROJECT_SOURCE_DIR}/src/external/thread_mpi/include)
target_compile_definitions(modularsimulator PRIVATE TMPI_USE_VISIBILITY)

# Source files have the following private external dependencies.
target_link_libraries(modularsimulator PRIVATE tng_io)

# Source files have the following private infrastructure dependencies.
target_link_libraries(modularsimulator PRIVATE common)

# Source files have the following private module dependencies.
# TODO: Explicitly link specific modules.
target_link_libraries(modularsimulator PRIVATE
                      legacy_api
                      legacy_modules
                      gmxlib
                      math
                      mdtypes
                      pbcutil
                      pulling
                      timing
                      tng_io
                      topology
                      utility
)

# Public interface for modules, including dependencies and interfaces
target_include_directories(modularsimulator PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if (BUILD_SHARED_LIBS)
    set_target_properties(modularsimulator PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
