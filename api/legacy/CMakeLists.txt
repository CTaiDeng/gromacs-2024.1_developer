# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2020- The GROMACS Authors
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

add_library(legacy_api INTERFACE)
target_include_directories(legacy_api INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

configure_file(version.h.cmakein include/gromacs/version.h)

# Allow an export target for legacy_api since it is a dependency of the
# installable libgromacs target.
install(TARGETS legacy_api
        EXPORT libgromacs
        INCLUDES DESTINATION include)

if(GMX_INSTALL_LEGACY_API)
    # Install public header directories.
    install(DIRECTORY include/gromacs
            DESTINATION include)

    # Install "configured" files from the build tree.
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/gromacs/version.h
            DESTINATION include/gromacs)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/gromacs/libgromacs_export.h
            DESTINATION include/gromacs)
endif()
