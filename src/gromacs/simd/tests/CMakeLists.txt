# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2014- The GROMACS Authors
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

gmx_add_unit_test(SimdUnitTests simd-test
    CPP_SOURCE_FILES
        base.cpp
        bootstrap_loadstore.cpp
        scalar.cpp
        scalar_util.cpp
        scalar_math.cpp
        simd.cpp
        simd_floatingpoint.cpp
        simd_floatingpoint_util.cpp
        simd_integer.cpp
        simd_math.cpp
        simd_memory.cpp
        simd_vector_operations.cpp
        simd4.cpp
        simd4_floatingpoint.cpp
        simd4_math.cpp
        simd4_vector_operations.cpp
        )
target_link_libraries(simd-test PRIVATE simd)
