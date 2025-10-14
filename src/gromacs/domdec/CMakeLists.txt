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

add_library(domdec INTERFACE)
file(GLOB DOMDEC_SOURCES *.cpp)
list(FILTER DOMDEC_SOURCES EXCLUDE REGEX ".*/gpuhaloexchange_impl_gpu[a-z_]*\.cpp$")

if(GMX_GPU_SYCL)
    file(GLOB DOMDEC_GPU_SOURCES gpuhaloexchange_impl_gpu.cpp gpuhaloexchange_impl_gpu_sycl.cpp)
    _gmx_add_files_to_property(SYCL_SOURCES gpuhaloexchange_impl_gpu.cpp gpuhaloexchange_impl_gpu_sycl.cpp)
endif()
if(GMX_GPU_CUDA)
  file(GLOB DOMDEC_GPU_SOURCES gpuhaloexchange_impl_gpu.cpp gpuhaloexchange_impl_gpu.cu)
  _gmx_add_files_to_property(CUDA_SOURCES gpuhaloexchange_impl_gpu.cpp)
endif()

set(LIBGROMACS_SOURCES ${LIBGROMACS_SOURCES} ${DOMDEC_SOURCES} ${DOMDEC_GPU_SOURCES} PARENT_SCOPE)

# Source files have the following dependencies on library infrastructure.
#target_link_libraries(domdec PRIVATE
#                      common
#                      legacy_modules
#)

# Public interface for modules, including dependencies and interfaces
#target_include_directories(domdec PUBLIC
target_include_directories(domdec INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(domdec PUBLIC
target_link_libraries(domdec INTERFACE
                      legacy_api
                      gpu_utils
                      mdlib
                      timing
                      topology
                      utility
                      )

# TODO: when domdec is an OBJECT target
#target_link_libraries(domdec PUBLIC legacy_api)
#target_link_libraries(domdec PRIVATE common)

# Source files have the following private module dependencies.
#target_link_libraries(domdec PRIVATE NOTHING)
# TODO: Explicitly link specific modules.
#target_link_libraries(domdec PRIVATE legacy_modules)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
