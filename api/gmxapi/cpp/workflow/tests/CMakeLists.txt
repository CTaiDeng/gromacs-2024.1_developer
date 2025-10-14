# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2018- The GROMACS Authors
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

gmx_add_gtest_executable(workflow-details-test
    CPP_SOURCE_FILES
        workflow.cpp
        # pseudo-library for code for mdrun
        $<TARGET_OBJECTS:mdrun_objlib>
)

target_link_libraries(workflow-details-test
                      PRIVATE
                        Gromacs::gmxapi
                        gmxapi-detail
                        gmxapi-testsupport
                        mdrun_test_infrastructure)

gmx_register_gtest_test(GmxapiInternalInterfaceTests workflow-details-test OPENMP_THREADS 2 INTEGRATION_TEST IGNORE_LEAKS QUICK_GPU_TEST)

set_tests_properties(GmxapiInternalInterfaceTests PROPERTIES
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

#
# Test with MPI.
#
# gmxapi is expected to work whether or not GROMACS was built with MPI (or tMPI)
# but we don't assume that MPI is available to the build tree or testing tools
# unless CMake was configured with MPI.
#
# For MPI-enabled testing of gmxapi clients using non-MPI-enabled GROMACS, we
# defer testing to the Python gmxapi client package tests.
#
if (GMX_MPI)
    gmx_add_gtest_executable(workflow-details-mpi-test MPI
                             CPP_SOURCE_FILES
                             workflow.cpp
                             # pseudo-library for code for mdrun
                             $<TARGET_OBJECTS:mdrun_objlib>
                             )

    target_link_libraries(workflow-details-mpi-test
                          PRIVATE
                          Gromacs::gmxapi
                          gmxapi-detail
                          gmxapi-testsupport
                          mdrun_test_infrastructure)

    gmx_register_gtest_test(GmxapiInternalsMpiTests workflow-details-mpi-test MPI_RANKS 2 OPENMP_THREADS 2 INTEGRATION_TEST IGNORE_LEAKS)

    if (TEST GmxapiInternalsMpiTests AND GMX_CAN_RUN_MPI_TESTS)
        set_tests_properties(GmxapiInternalsMpiTests PROPERTIES
                             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif ()
endif ()
