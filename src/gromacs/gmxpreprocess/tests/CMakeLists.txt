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

gmx_add_gtest_executable(gmxpreprocess-test
    CPP_SOURCE_FILES
        editconf.cpp
        genconf.cpp
        genion.cpp
        genrestr.cpp
        gen_maxwell_velocities.cpp
        gpp_atomtype.cpp
        gpp_bond_atomtype.cpp
        grompp_directives.cpp
        insert_molecules.cpp
        massrepartitioning.cpp
        readir.cpp
        solvate.cpp
        topdirs.cpp
        )
gmx_register_gtest_test(GmxPreprocessTests gmxpreprocess-test SLOW_TEST)
target_link_libraries(gmxpreprocess-test PRIVATE
        gmxpreprocess
        topology
        )

# Currently these can be slow to run in CI, so they are in
# several test binaries.

set(exename pdb2gmx1-test)
gmx_add_gtest_executable(${exename}
    CPP_SOURCE_FILES
       pdb2gmx.cpp
       )
target_compile_definitions(${exename} PRIVATE OPLSAA=1 GROMOS=0 AMBER=0 CHARMM=0)
gmx_register_gtest_test(Pdb2gmx1Test ${exename} SLOW_TEST)

set(exename pdb2gmx2-test)
gmx_add_gtest_executable(${exename}
    CPP_SOURCE_FILES
        pdb2gmx.cpp
        )
target_compile_definitions(${exename} PRIVATE OPLSAA=0 GROMOS=1 AMBER=0 CHARMM=0)
gmx_register_gtest_test(Pdb2gmx2Test ${exename} SLOW_TEST)

set(exename pdb2gmx3-test)
gmx_add_gtest_executable(${exename}
    CPP_SOURCE_FILES
        pdb2gmx.cpp
        )
target_compile_definitions(${exename} PRIVATE OPLSAA=0 GROMOS=0 AMBER=1 CHARMM=1)
gmx_register_gtest_test(Pdb2gmx3Test ${exename} SLOW_TEST)
