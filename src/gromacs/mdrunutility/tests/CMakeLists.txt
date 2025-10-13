# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2016- The GROMACS Authors
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

gmx_add_unit_test_library(mdrunutility-test-shared
                          threadaffinitytest.cpp)
target_link_libraries(mdrunutility-test-shared PRIVATE common legacy_api)

gmx_add_unit_test(MdrunUtilityUnitTests mdrunutility-test
    CPP_SOURCE_FILES
        mdmodulesnotifier.cpp
        threadaffinity.cpp
        )
target_link_libraries(mdrunutility-test PRIVATE mdrunutility-test-shared)

gmx_add_mpi_unit_test(MdrunUtilityMpiUnitTests mdrunutility-mpi-test 4
    CPP_SOURCE_FILES
        threadaffinity_mpi.cpp
        )
# TODO This target does not exist in a non-MPI build. It would be preferable
# to change functions like gmx_add_mpi_unit_test to add link-time dependencies
# as well as source files.
if (GMX_MPI OR (GMX_THREAD_MPI AND GTEST_IS_THREADSAFE))
    target_link_libraries(mdrunutility-mpi-test PRIVATE mdrunutility-test-shared)
endif()
