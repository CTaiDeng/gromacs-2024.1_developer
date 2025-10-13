# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2012- The GROMACS Authors
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

macro(get_compiler_info LANGUAGE BUILD_COMPILER)
    if(GMX_INTEL_LLVM AND NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        # Manually set compiler info for Intel LLVM. Can be removed after we require CMake 3.20+.
        set(${BUILD_COMPILER} "${CMAKE_${LANGUAGE}_COMPILER} IntelLLVM ${GMX_INTEL_LLVM_VERSION}")
    else()
        set(${BUILD_COMPILER} "${CMAKE_${LANGUAGE}_COMPILER} ${CMAKE_${LANGUAGE}_COMPILER_ID} ${CMAKE_${LANGUAGE}_COMPILER_VERSION}")
    endif()
endmacro()
