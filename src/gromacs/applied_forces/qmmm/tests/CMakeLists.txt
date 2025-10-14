# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2021- The GROMACS Authors
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

gmx_add_unit_test(QMMMAppliedForcesUnitTest qmmm_applied_forces-test
    CPP_SOURCE_FILES
        qmmminputgenerator.cpp
        qmmmtopologypreprocessor.cpp
        qmmmoptions.cpp
        qmmmforceprovider.cpp
        qmmm.cpp
        )
target_link_libraries(qmmm_applied_forces-test PRIVATE math)

# For QMMMForceProvider test we need to define wether libcp2k linked
if (GMX_CP2K)
    set_source_files_properties(qmmmforceprovider.cpp PROPERTIES COMPILE_DEFINITIONS GMX_CP2K=1)
else()
    set_source_files_properties(qmmmforceprovider.cpp PROPERTIES COMPILE_DEFINITIONS GMX_CP2K=0)
endif()

target_link_libraries(qmmm_applied_forces-test PRIVATE topology)
