# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2010- The GROMACS Authors
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

gmx_add_gtest_executable(trajectoryanalysis-test
    CPP_SOURCE_FILES
        # Infrastructure
        moduletest.cpp
        # Tests
        angle.cpp
        clustsize.cpp
        cmdlinerunner.cpp
        convert_trj.cpp
        distance.cpp
        dssp.cpp
        extract_cluster.cpp
        freevolume.cpp
        gyrate.cpp
        hbond.cpp
        msd.cpp
        pairdist.cpp
        rdf.cpp
        sasa.cpp
        select.cpp
        scattering.cpp
        surfacearea.cpp
        topologyinformation.cpp
        trajectory.cpp
        unionfind.cpp
        $<TARGET_OBJECTS:mdrun_objlib>
        )
gmx_register_gtest_test(TrajectoryAnalysisUnitTests trajectoryanalysis-test SLOW_TEST)
target_link_libraries(trajectoryanalysis-test PRIVATE analysisdata-test-shared)
target_link_libraries(trajectoryanalysis-test PRIVATE common)

add_executable(test_selection ${UNITTEST_TARGET_OPTIONS} test_selection.cpp)
target_link_libraries(test_selection PRIVATE
                      common
                      libgromacs
                      ${GMX_EXE_LINKER_FLAGS})
# TODO: Explicitly link to module dependencies.
target_link_libraries(test_selection PRIVATE legacy_modules utility)
