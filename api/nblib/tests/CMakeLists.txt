# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2020- The GROMACS Authors
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

gmx_add_unit_test_library(nblib_test_infrastructure
        testsystems.cpp
    )
target_include_directories(nblib_test_infrastructure SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/src/external)
target_include_directories(nblib_test_infrastructure PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nblib_test_infrastructure PUBLIC
    nblib
    legacy_api
    math
    pbcutil
    simd
    topology
    )

set(testname "NbLibSetupTests")
set(exename "nblib-setup-test")

gmx_add_gtest_executable(
    ${exename}
    HARDWARE_DETECTION
    CPP_SOURCE_FILES
    # files with code for tests
        box.cpp
        interactions.cpp
        particletype.cpp
        pbcholder.cpp
        molecules.cpp
        nbnxmsetup.cpp
        topology.cpp
        virials.cpp
    )
target_link_libraries(${exename} PRIVATE mdrun_test_infrastructure)
target_link_libraries(${exename} PRIVATE nblib_test_infrastructure nblib)
target_include_directories(${exename} PRIVATE ${PROJECT_SOURCE_DIR}/api)
gmx_register_gtest_test(${testname} ${exename} INTEGRATION_TEST)
add_dependencies(nblib-tests ${exename})

set(exename "nblib-tpr-test")

gmx_add_unit_test(
        NbLibTprTests
        ${exename}
        CPP_SOURCE_FILES
        # files with code for tests
        tpr.cpp
)
target_link_libraries(${exename} PRIVATE mdrun_test_infrastructure)
target_link_libraries(${exename} PRIVATE nblib_test_infrastructure nblib)
add_dependencies(nblib-tests ${exename})

set(testname "NbLibIntegrationTests")
set(exename "nblib-integration-test")

gmx_add_gtest_executable(
    ${exename}
    CPP_SOURCE_FILES
    # files with code for tests
        gmxcalculator.cpp
        nbkernelsystem.cpp
        simstate.cpp
    )
target_link_libraries(${exename} PRIVATE nblib_test_infrastructure nblib)
gmx_register_gtest_test(${testname} ${exename} INTEGRATION_TEST)
add_dependencies(nblib-tests ${exename})

# The integrator sometimes times out on TSAN so it gets its own test harness
set(testname "NbLibIntegratorTests")
set(exename "nblib-integrator-test")

gmx_add_gtest_executable(
        ${exename}
        CPP_SOURCE_FILES
        integrator.cpp
)
target_link_libraries(${exename} PRIVATE nblib_test_infrastructure nblib)
gmx_register_gtest_test(${testname} ${exename} INTEGRATION_TEST)
add_dependencies(nblib-tests ${exename})
