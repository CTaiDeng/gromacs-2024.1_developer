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

gmx_add_gtest_executable(fft-test HARDWARE_DETECTION
    GPU_CPP_SOURCE_FILES
        fft.cpp
    )
gmx_register_gtest_test(FFTUnitTests fft-test SLOW_TEST QUICK_GPU_TEST)
target_link_libraries(
    fft-test PRIVATE
        fft
        gpu_utils
        testutils
        utility
)

if(GMX_USE_Heffte OR GMX_USE_cuFFTMp)
gmx_add_mpi_unit_test(FFTMpiUnitTests fft-mpi-test 4 HARDWARE_DETECTION
    GPU_CPP_SOURCE_FILES
        fft_mpi.cpp
        )
target_link_libraries(fft-mpi-test PRIVATE fft)
endif()

if (TARGET fft-mpi-test)
    target_link_libraries(
        fft-mpi-test PRIVATE
            fft
            gpu_utils
            testutils
            utility
    )
endif ()
