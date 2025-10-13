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

gmx_add_unit_test(MdlibUnitTest mdlib-test HARDWARE_DETECTION
    CPP_SOURCE_FILES
        calc_verletbuf.cpp
        calcvir.cpp
        constr.cpp
        constrtestdata.cpp
        constrtestrunners.cpp
        ebin.cpp
        energydrifttracker.cpp
        energyoutput.cpp
        expanded.cpp
        freeenergyparameters.cpp
        leapfrog.cpp
        leapfrogtestdata.cpp
        leapfrogtestrunners.cpp
        parrinellorahman.cpp
        settle.cpp
        settletestdata.cpp
        settletestrunners.cpp
        shake.cpp
        simulationsignal.cpp
        updategroups.cpp
        updategroupscog.cpp
    GPU_CPP_SOURCE_FILES
        constrtestrunners_gpu.cpp
        leapfrogtestrunners_gpu.cpp
        mdgpugraph.cpp
        settletestrunners_gpu.cpp
        wholemoleculetransform.cpp
        )
target_link_libraries(mdlib-test PRIVATE
        mdlib
        math
        )
