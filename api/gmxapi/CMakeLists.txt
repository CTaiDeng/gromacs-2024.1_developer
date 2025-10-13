# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2018- The GROMACS Authors
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

set(GMXAPI_MAJOR 0)
set(GMXAPI_MINOR 4)
set(GMXAPI_PATCH 0)
set(GMXAPI_RELEASE ${GMXAPI_MAJOR}.${GMXAPI_MINOR}.${GMXAPI_PATCH})

add_library(gmxapi)
set_target_properties(gmxapi PROPERTIES
                      VERSION_MAJOR ${GMXAPI_MAJOR}
                      VERSION_MINOR ${GMXAPI_MINOR}
                      VERSION_PATCH ${GMXAPI_PATCH}
                      RELEASE ${GMXAPI_RELEASE})


if (GMX_LIB_MPI)
    # GROMACS is built against an MPI library.
    set(_gmx_mpi_type "library")
    target_link_libraries(gmxapi PRIVATE MPI::MPI_CXX)
    # a660b04e8d8756ed9c4dd355ad5d0d23d503eaa9 removed the definition for _gmxapi_find_dependencies
    # that was used in gmxapi-config.cmake.in because we do not intend to require clients to build
    # with MPI support. However, we _should_ provide some sort of helpers for clients to find
    # tool chain and MPI installation consistent with what would be needed. We could insert a
    # find_dependency for MPI, preconfigured with our best guess of hints. It would probably be better,
    # though, to add the appropriate hints to the gromacs-hints.cmake so that the client can easily
    # set or override input variables arbitrarily early in their CMake config call.
    # _gmxapi_find_dependencies is now set in api/gmxapi/cpp/CMakeLists.txt
elseif(GMX_THREAD_MPI)
    # GROMACS is built with its internal thread-MPI implementation.
    set(_gmx_mpi_type "tmpi")
else()
    # GROMACS is built without MPI or thread-MPI.
    set(_gmx_mpi_type "none")
endif ()
define_property(TARGET PROPERTY MPI
                BRIEF_DOCS "MPI capability of the GROMACS library."
                FULL_DOCS "Values of 'library', 'tmpi', or 'none' indicate the configure-time options of the GROMACS
                 library build.")
set_target_properties(gmxapi PROPERTIES
                      MPI ${_gmx_mpi_type})
unset(_gmx_mpi_type)

##########################
# Define public interface.
#
# The include directory should be mostly empty so that we can use it internally as
# the public interface include directory during build and testing.
configure_file(include/gmxapiversion.h.in include/gmxapi/version.h)
configure_file(include/resourceassignment.cmakein.h include/gmxapi/mpi/resourceassignment.h)
target_include_directories(gmxapi PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
                           $<INSTALL_INTERFACE:include>
                           )

add_subdirectory(cpp)

###############################
# Install the public interface.
#
# If any item begins in a generator expression it must evaluate to a full path,
# so we can't just use something like $<TARGET_PROPERTIES:gmxapiPublicHeaders,SOURCES>.
# Instead, we use a canonical list defined in the parent scope.

# Install public header directories.
install(DIRECTORY include/gmxapi
        DESTINATION include)

# Install "configured" files from the build tree.
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/gmxapi/version.h
        DESTINATION include/gmxapi)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/gmxapi/mpi/resourceassignment.h
        DESTINATION include/gmxapi/mpi)
