# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2010- The GROMACS Authors
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

add_library(selection INTERFACE)

file(GLOB SELECTION_SOURCES *.cpp)
file(GLOB SCANNER_SOURCES scanner.cpp parser.cpp)
list(REMOVE_ITEM SELECTION_SOURCES ${SCANNER_SOURCES})
# Add the non-generated sources to libgromacs, so that they have only the normal warning suppressions
target_sources(libgromacs PRIVATE ${SELECTION_SOURCES})

add_library(scanner OBJECT ${SCANNER_SOURCES})
gmx_target_compile_options(scanner)
target_compile_definitions(scanner PRIVATE HAVE_CONFIG_H)
# Compilers give lots of warnings on these generated files.  These
# are awkward to suppress inline, so we do it in the compilation
# command (after testing that the compiler supports the
# suppressions).
if (WIN32)
    gmx_target_warning_suppression(scanner /wd4101 HAS_NO_MSVC_UNUSED)
    gmx_target_warning_suppression(scanner /wd4244 HAS_NO_MSVC_LOSSY_CONVERSION)
    gmx_target_warning_suppression(scanner /wd4267 HAS_NO_MSVC_LOSSY_CONVERSION_SIZE_T_TO_INT)
else()
    gmx_target_warning_suppression(scanner -Wno-unused HAS_NO_UNUSED)
    gmx_target_warning_suppression(scanner -Wno-unused-parameter HAS_NO_UNUSED_PARAMETER)
    gmx_target_warning_suppression(scanner -Wno-missing-declarations HAS_NO_MISSING_DECLARATIONS)
    gmx_target_warning_suppression(scanner -Wno-null-conversion HAS_NO_NULL_CONVERSIONS)
endif()
target_link_libraries(scanner PRIVATE legacy_api)
# TODO: Remove `legacy_modules` once specific modules are explicitly linked.
target_link_libraries(scanner PRIVATE legacy_modules)
target_link_libraries(scanner PRIVATE utility)

list(APPEND libgromacs_object_library_dependencies scanner)
set(libgromacs_object_library_dependencies ${libgromacs_object_library_dependencies} PARENT_SCOPE)

# Source files have the following dependencies on library infrastructure.
#target_link_libraries(selection PRIVATE
#                      common
#                      legacy_modules
#)

# Public interface for modules, including dependencies and interfaces
#target_include_directories(selection PUBLIC
target_include_directories(selection INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(selection PUBLIC
target_link_libraries(selection INTERFACE
                      legacy_api
                      )

# TODO: when selection is an OBJECT target
#target_link_libraries(selection PUBLIC legacy_api)
#target_link_libraries(selection PRIVATE common)

# Source files have the following private module dependencies.
#target_link_libraries(selection PRIVATE NOTHING)
# TODO: Explicitly link specific modules.
#target_link_libraries(selection PRIVATE legacy_modules)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
