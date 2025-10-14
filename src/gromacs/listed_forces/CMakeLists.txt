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

add_library(listed_forces INTERFACE)
gmx_add_libgromacs_sources(
    bonded.cpp
    disre.cpp
    listed_forces_gpu_impl.cpp
    listed_forces.cpp
    listed_internal.cpp
    manage_threading.cpp
    orires.cpp
    pairs.cpp
    position_restraints.cpp
    restcbt.cpp
    )

if(GMX_GPU_CUDA)
    gmx_add_libgromacs_sources(
       listed_forces_gpu_impl_gpu.cpp
       listed_forces_gpu_internal.cu
       )
    _gmx_add_files_to_property(CUDA_SOURCES
      listed_forces_gpu_impl_gpu.cpp
      )
endif()

if (GMX_GPU_SYCL)
    gmx_add_libgromacs_sources(
        listed_forces_gpu_impl_gpu.cpp
        listed_forces_gpu_internal_sycl.cpp
        )
    _gmx_add_files_to_property(SYCL_SOURCES
        listed_forces_gpu_impl_gpu.cpp
        listed_forces_gpu_internal_sycl.cpp
        )
endif()

# Source files have the following private module dependencies.
target_link_libraries(listed_forces PRIVATE
#                      gmxlib
#                      math
#                      mdtypes
                      )

# Public interface for modules, including dependencies and interfaces
#target_include_directories(listed_forces PUBLIC
target_include_directories(listed_forces INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(listed_forces PUBLIC
target_link_libraries(listed_forces INTERFACE
                      legacy_api
                      )

# TODO: when listed_forces is an OBJECT target
#target_link_libraries(listed_forces PUBLIC legacy_api)
#target_link_libraries(listed_forces PRIVATE common)

# Source files have the following private module dependencies.
#target_link_libraries(listed_forces PRIVATE tng_io)
# TODO: Explicitly link specific modules.
#target_link_libraries(listed_forces PRIVATE legacy_modules)

if (BUILD_TESTING)
     add_subdirectory(tests)
endif()
