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

add_library(mdtypes INTERFACE)

file(GLOB MDTYPES_SOURCES
    awh_history.cpp
    checkpointdata.cpp
    commrec.cpp
    df_history.cpp
    energyhistory.cpp
    forcebuffers.cpp
    group.cpp
    iforceprovider.cpp
    inputrec.cpp
    interaction_const.cpp
    md_enums.cpp
    multipletimestepping.cpp
    observableshistory.cpp
    observablesreducer.cpp
    state.cpp
    threaded_force_buffer.cpp)

if(GMX_GPU)
    gmx_add_libgromacs_sources(
       state_propagator_data_gpu_impl_gpu.cpp
       )
   if(GMX_GPU_CUDA)
       _gmx_add_files_to_property(CUDA_SOURCES
           state_propagator_data_gpu_impl_gpu.cpp
           )
   endif()
   if(GMX_GPU_SYCL)
       _gmx_add_files_to_property(SYCL_SOURCES
           state_propagator_data_gpu_impl_gpu.cpp
           )
   endif()
else()
    gmx_add_libgromacs_sources(
      state_propagator_data_gpu_impl.cpp
      )
endif()

# Source files have the following private module dependencies.
target_link_libraries(mdtypes PRIVATE
                      #                      gmxlib
                      #                      math
                      #                      mdtypes
                      #                      tng_io
                      )

# Public interface for modules, including dependencies and interfaces
#target_include_directories(mdtypes PUBLIC
target_include_directories(mdtypes INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(mdtypes PUBLIC
target_link_libraries(mdtypes INTERFACE
                      legacy_api
                      gpu_utils
                      math
                      )
# Interface conveys transitive dependence on other interface(s)
target_link_libraries(mdtypes INTERFACE utility)

# TODO: when fileio is an OBJECT target
#target_link_libraries(mdtypes PUBLIC legacy_api)
#target_link_libraries(mdtypes PRIVATE common)

# Source files have the following private module dependencies.
#target_link_libraries(mdtypes PRIVATE tng_io)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(LIBGROMACS_SOURCES ${LIBGROMACS_SOURCES} ${MDTYPES_SOURCES} PARENT_SCOPE)
