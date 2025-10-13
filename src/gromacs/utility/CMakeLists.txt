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

add_library(utility INTERFACE)

file(GLOB UTILITY_SOURCES *.cpp)
if (GMX_GPU_CUDA)
    set_source_files_properties(cuda_version_information.cu PROPERTIES LANGUAGE CUDA)
    set_property(GLOBAL APPEND PROPERTY GMX_LIBGROMACS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version_information.cu)
endif ()
if (GMX_GPU_SYCL)
    _gmx_add_files_to_property(SYCL_SOURCES
                               sycl_version_information.cpp
                               )
endif ()
set(LIBGROMACS_SOURCES ${LIBGROMACS_SOURCES} ${UTILITY_SOURCES} PARENT_SCOPE)

# Public interface for modules, including dependencies and interfaces
#target_include_directories(utility PUBLIC
target_include_directories(utility INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(utility PUBLIC
target_link_libraries(utility INTERFACE
                      legacy_api
                      )
if (GMX_LIB_MPI)
    target_link_libraries(utility INTERFACE MPI::MPI_CXX)
else ()
    target_include_directories(
        utility INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/external/thread_mpi/include>
    )
    target_compile_definitions(utility INTERFACE TMPI_USE_VISIBILITY)
endif ()

# TODO(#3288): when utility is an OBJECT target, explicitly link source dependencies.
# Source files have the following private module dependencies.
#target_link_libraries(utility PRIVATE common)
# TODO(#3288): Explicitly link specific modules as legacy_modules is removed.
#target_link_libraries(utility PRIVATE legacy_modules)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif ()
