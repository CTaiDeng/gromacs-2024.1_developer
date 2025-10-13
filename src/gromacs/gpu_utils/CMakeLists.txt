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

add_library(gpu_utils INTERFACE)

gmx_add_libgromacs_sources(
        clfftinitializer.cpp
        device_stream_manager.cpp
        hostallocator.cpp
        gpu_utils.cpp
        gpueventsynchronizer_helpers.cpp
        nvshmem_utils.cpp
        )
if(GMX_GPU_OPENCL)
    gmx_add_libgromacs_sources(
        device_context_ocl.cpp
        device_stream_ocl.cpp
        pmalloc.cpp
        ocl_compiler.cpp
        ocl_caching.cpp
        oclutils.cpp
        )
elseif(GMX_GPU_CUDA)
    gmx_add_libgromacs_sources(
        device_context.cpp
        device_stream.cu
        gpu_utils.cu
        pmalloc.cu
        )
    _gmx_add_files_to_property(CUDA_SOURCES
        device_stream_manager.cpp
        )
elseif(GMX_GPU_SYCL)
    gmx_add_libgromacs_sources(
        devicebuffer_sycl.cpp
        device_context_sycl.cpp
        device_stream_sycl.cpp
        pmalloc_sycl.cpp
        )
    _gmx_add_files_to_property(SYCL_SOURCES
        devicebuffer_sycl.cpp
        device_context_manager.cpp
        device_context_sycl.cpp
        device_stream_manager.cpp
        device_stream_sycl.cpp
        pmalloc_sycl.cpp
        )
else()
    gmx_add_libgromacs_sources(
        device_context.cpp
        device_stream.cpp
        pmalloc.cpp
    )
endif()

# Source files have the following private module dependencies.
target_link_libraries(gpu_utils PRIVATE
#                      gmxlib
#                      math
#                      mdtypes
#                      tng_io
                      )

# Public interface for modules, including dependencies and interfaces
#target_include_directories(gpu_utils PUBLIC
target_include_directories(gpu_utils INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(gpu_utils PUBLIC
target_link_libraries(gpu_utils INTERFACE
                      legacy_api
                      math
                      )

# TODO: when gpu_utils is an OBJECT target
#target_link_libraries(gpu_utils PUBLIC legacy_api)
#target_link_libraries(gpu_utils PRIVATE common)

# Source files have the following private module dependencies.
#target_link_libraries(gpu_utils PRIVATE tng_io)
# TODO: Explicitly link specific modules.
#target_link_libraries(gpu_utils PRIVATE legacy_modules)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
