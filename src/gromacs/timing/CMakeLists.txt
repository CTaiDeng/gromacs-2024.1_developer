# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2013- The GROMACS Authors
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

add_library(timing INTERFACE)
set(TIMING_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cyclecounter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wallcycle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/walltime_accounting.cpp
    )
set(LIBGROMACS_SOURCES ${LIBGROMACS_SOURCES} ${TIMING_SOURCES} PARENT_SCOPE)

# Source files have the following dependencies on library infrastructure.
#target_link_libraries(timing PRIVATE
#                      common
#                      legacy_modules
#)

# Public interface for modules, including dependencies and interfaces
#target_include_directories(timing PUBLIC
target_include_directories(timing INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(timing PUBLIC
target_link_libraries(timing INTERFACE
                      legacy_api
                      )

# Tracing support for NVTX / ROCTX / ITT (with some basic includ/lib detection support,
# sufficient for most use-cases and this is a dev-feature anyway).
if(GMX_USE_NVTX)
    find_path (NVTX_INCLUDE_DIR
        NAMES nvToolsExt.h
        HINTS ENV CUDA_HOME "${CUDA_TOOLKIT_ROOT_DIR}"
        PATH_SUFFIXES include include/nvtx3
        REQUIRED
        )
    find_library (NVTX_LIBRARY
        NAMES libnvToolsExt.so
        HINTS ENV CUDA_HOME "${CUDA_TOOLKIT_ROOT_DIR}"
        PATH_SUFFIXES lib64 lib
        REQUIRED
        )
    target_include_directories(timing INTERFACE "${NVTX_INCLUDE_DIR}")
    target_link_libraries(timing INTERFACE "${NVTX_LIBRARY}")
    # As of CUDA 11.8, there are a lot of old-style casts in nvToolsExt.h
    gmx_target_interface_warning_suppression(timing "-Wno-old-style-cast" HAS_WARNING_NO_OLD_STYLE_CAST)
endif()

if(GMX_USE_ROCTX)
    find_path (ROCTX_INCLUDE_DIR
        NAMES roctracer/roctx.h
        HINTS ENV ROCM_PATH ENV ROCM_HOME "${HIPSYCL_SYCLCC_ROCM_PATH}"
        PATH_SUFFIXES include
        REQUIRED
        )
    find_library (ROCTX_LIBRARY
        NAMES libroctx64.so
        HINTS ENV ROCM_PATH ENV ROCM_HOME "${HIPSYCL_SYCLCC_ROCM_PATH}"
        PATH_SUFFIXES roctracer/lib64 roctracer/lib lib64 lib
        REQUIRED
        )
    target_include_directories(timing INTERFACE "${ROCTX_INCLUDE_DIR}")
    target_link_libraries(timing INTERFACE "${ROCTX_LIBRARY}")
endif()

if(GMX_USE_ITT)
    find_path (ITTNOTIFY_INCLUDE_DIR
        NAMES ittnotify.h
        HINTS ENV VTUNE_PROFILER_DIR
        PATH_SUFFIXES include
        REQUIRED
        )
    find_library (ITTNOTIFY_LIBRARY
        NAMES libittnotify.a # We need the static library
        HINTS ENV VTUNE_PROFILER_DIR
        PATH_SUFFIXES lib64 lib
        REQUIRED
        )
    target_include_directories(timing INTERFACE "${ITTNOTIFY_INCLUDE_DIR}" SYSTEM)
    target_link_libraries(timing INTERFACE "${ITTNOTIFY_LIBRARY}")
endif()

# TODO: when timing is an OBJECT target
#target_link_libraries(timing PUBLIC legacy_api)
#target_link_libraries(timing PRIVATE common)

# Source files have the following private module dependencies.
#target_link_libraries(timing PRIVATE NOTHING)
# TODO: Explicitly link specific modules.
#target_link_libraries(timing PRIVATE legacy_modules)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
