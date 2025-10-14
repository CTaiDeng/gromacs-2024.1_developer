# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2016- The GROMACS Authors
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

if (GMX_GPU_FFT_VKFFT)
  # Suppress warning that first appeared when compiling tests on macOS with x86.
  check_cxx_compiler_flag("-Wno-unused-parameter" HAS_NO_UNUSED_PARAMETER)
  if(HAS_NO_UNUSED_PARAMETER)
    set_source_files_properties(pmetestcommon.cpp PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")
  endif()
endif()

gmx_add_unit_test(EwaldUnitTests ewald-test
    HARDWARE_DETECTION
    DYNAMIC_REGISTRATION
    CPP_SOURCE_FILES
        pmebsplinetest.cpp
        pmegathertest.cpp
        pmesolvetest.cpp
        pmesplinespreadtest.cpp
        pmetestcommon.cpp
        pme.cpp
)
target_link_libraries(
    ewald-test PRIVATE
        ewald
        domdec
        fft
        gpu_utils
        math
        mdtypes
        pbcutil
        testutils
        topology
        utility
)
