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

set(testname "NbLibListedForcesTests")
set(exename "nblib-listed-forces-test")

gmx_add_gtest_executable(
        ${exename}
        CPP_SOURCE_FILES
        # files with code for tests
        bondtypes.cpp
        gmxcalculator.cpp
        helpers.cpp
        listedtesthelpers.cpp
        kernels.cpp
        typetests.cpp
        calculator.cpp
        conversions.cpp
        shiftforces.cpp
        transformations.cpp
)
target_link_libraries(${exename} PRIVATE nblib_test_infrastructure nblib
                      listed_forces
                      pbcutil
                      timing)
gmx_register_gtest_test(${testname} ${exename} INTEGRATION_TEST)
add_dependencies(check-nblib ${exename})
