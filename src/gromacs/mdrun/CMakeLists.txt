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

add_library(mdrun INTERFACE)
gmx_add_libgromacs_sources(
    legacymdrunoptions.cpp
    legacysimulator.cpp
    md.cpp
    mdmodules.cpp
    membedholder.cpp
    mimic.cpp
    minimize.cpp
    replicaexchange.cpp
    rerun.cpp
    runner.cpp
    shellfc.cpp
    simulationcontext.cpp
    simulationinput.cpp
    simulationinputhandle.cpp
    simulatorbuilder.cpp
    tpi.cpp
    )

# Source files have the following private module dependencies.
target_link_libraries(mdrun PRIVATE
#                      gmxlib
#                      math
#                      mdtypes
#                      tng_io
                      )

# Public interface for modules, including dependencies and interfaces
#target_include_directories(mdrun PUBLIC
target_include_directories(mdrun INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#target_link_libraries(mdrun PUBLIC
target_link_libraries(mdrun INTERFACE
                      legacy_api
                      )

# TODO: when mdrun is an OBJECT target
#target_link_libraries(mdrun PUBLIC legacy_api)
#target_link_libraries(mdrun PRIVATE common)

# Source files have the following private module dependencies.
#target_link_libraries(mdrun PRIVATE tng_io)
# TODO: Explicitly link specific modules.
#target_link_libraries(mdrun PRIVATE legacy_modules)

# TODO: Find a home for this header and a scheme for installation.
# This header straddles the installed libraries and is a transitive interface
# from libgromacs to libgmxapi to libgmxapi clients. Near term efforts are
# planned to consolidate these libraries and refine the public interface.
# In the mean time, this is kept out of the gmxapi to avoid circular dependencies
# between the libraries. It can be moved to a separate interface-only target,
# but that is beyond the scope of issue #3379, which introduces the header.
# Ref #3152 and #3652
install(FILES simulationinputhandle.h
        DESTINATION include/gromacs/mdrun)

if (BUILD_TESTING)
# TODO import this from src/programs/mdrun/tests
#    add_subdirectory(tests)
endif()
