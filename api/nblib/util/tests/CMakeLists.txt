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

set(testname "NbLibUtilTests")
set(exename "nblib-util-test")

gmx_add_gtest_executable(
        ${exename}
        CPP_SOURCE_FILES
        setup.cpp
        traits.cpp
)
target_link_libraries(${exename} PRIVATE nblib_test_infrastructure nblib)
gmx_register_gtest_test(${testname} ${exename})
add_dependencies(check-nblib ${exename})
