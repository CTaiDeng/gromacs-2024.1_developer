# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2021- The GROMACS Authors
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

SET(CMAKE_C_COMPILER  "@CMAKE_C_COMPILER@" CACHE FILEPATH "C compiler used for GROMACS.")
SET(CMAKE_CXX_COMPILER "@CMAKE_CXX_COMPILER@" CACHE FILEPATH "CXX compiler used for GROMACS.")
SET(CMAKE_LINKER "@CMAKE_LINKER@" CACHE FILEPATH "Linker used for GROMACS.")
SET(GMX_CMAKE_VERSION "@CMAKE_VERSION@" "Version of CMake used to build these CMake config files.")
@_gmx_mpi_config@
@_gmx_osx_config@
@_gmx_cuda_config@
@_gmx_hipsycl_config@
