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

gmx_add_unit_test(DomDecTests domdec-test
    CPP_SOURCE_FILES
        hashedmap.cpp
        localatomsetmanager.cpp
        )
target_link_libraries(domdec-test PRIVATE domdec)

target_link_libraries(
    domdec-test PRIVATE
        domdec
        gpu_utils
        mdtypes
        testutils
)

gmx_add_mpi_unit_test(DomDecMpiTests domdec-mpi-test 4 HARDWARE_DETECTION
    GPU_CPP_SOURCE_FILES
        haloexchange_mpi.cpp
        )

if (TARGET domdec-mpi-test)
    target_link_libraries(
        domdec-mpi-test PRIVATE
            domdec
            gpu_utils
            mdtypes
            testutils
    )
endif ()
