# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2020- The GROMACS Authors
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

if (GMX_USE_SIMD_KERNELS)
    file(GLOB KERNEL_SOURCES
        kernel_ElecEwTwinCut_VdwLJCombGeom_F.cpp
        kernel_ElecEwTwinCut_VdwLJCombGeom_VF.cpp
        kernel_ElecEwTwinCut_VdwLJCombGeom_VgrpF.cpp
        kernel_ElecEwTwinCut_VdwLJCombLB_F.cpp
        kernel_ElecEwTwinCut_VdwLJCombLB_VF.cpp
        kernel_ElecEwTwinCut_VdwLJCombLB_VgrpF.cpp
        kernel_ElecEwTwinCut_VdwLJEwCombGeom_F.cpp
        kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF.cpp
        kernel_ElecEwTwinCut_VdwLJEwCombGeom_VgrpF.cpp
        kernel_ElecEwTwinCut_VdwLJ_F.cpp
        kernel_ElecEwTwinCut_VdwLJFSw_F.cpp
        kernel_ElecEwTwinCut_VdwLJFSw_VF.cpp
        kernel_ElecEwTwinCut_VdwLJFSw_VgrpF.cpp
        kernel_ElecEwTwinCut_VdwLJPSw_F.cpp
        kernel_ElecEwTwinCut_VdwLJPSw_VF.cpp
        kernel_ElecEwTwinCut_VdwLJPSw_VgrpF.cpp
        kernel_ElecEwTwinCut_VdwLJ_VF.cpp
        kernel_ElecEwTwinCut_VdwLJ_VgrpF.cpp
        kernel_ElecEw_VdwLJCombGeom_F.cpp
        kernel_ElecEw_VdwLJCombGeom_VF.cpp
        kernel_ElecEw_VdwLJCombGeom_VgrpF.cpp
        kernel_ElecEw_VdwLJCombLB_F.cpp
        kernel_ElecEw_VdwLJCombLB_VF.cpp
        kernel_ElecEw_VdwLJCombLB_VgrpF.cpp
        kernel_ElecEw_VdwLJEwCombGeom_F.cpp
        kernel_ElecEw_VdwLJEwCombGeom_VF.cpp
        kernel_ElecEw_VdwLJEwCombGeom_VgrpF.cpp
        kernel_ElecEw_VdwLJ_F.cpp
        kernel_ElecEw_VdwLJFSw_F.cpp
        kernel_ElecEw_VdwLJFSw_VF.cpp
        kernel_ElecEw_VdwLJFSw_VgrpF.cpp
        kernel_ElecEw_VdwLJPSw_F.cpp
        kernel_ElecEw_VdwLJPSw_VF.cpp
        kernel_ElecEw_VdwLJPSw_VgrpF.cpp
        kernel_ElecEw_VdwLJ_VF.cpp
        kernel_ElecEw_VdwLJ_VgrpF.cpp
        kernel_ElecQSTabTwinCut_VdwLJCombGeom_F.cpp
        kernel_ElecQSTabTwinCut_VdwLJCombGeom_VF.cpp
        kernel_ElecQSTabTwinCut_VdwLJCombGeom_VgrpF.cpp
        kernel_ElecQSTabTwinCut_VdwLJCombLB_F.cpp
        kernel_ElecQSTabTwinCut_VdwLJCombLB_VF.cpp
        kernel_ElecQSTabTwinCut_VdwLJCombLB_VgrpF.cpp
        kernel_ElecQSTabTwinCut_VdwLJEwCombGeom_F.cpp
        kernel_ElecQSTabTwinCut_VdwLJEwCombGeom_VF.cpp
        kernel_ElecQSTabTwinCut_VdwLJEwCombGeom_VgrpF.cpp
        kernel_ElecQSTabTwinCut_VdwLJ_F.cpp
        kernel_ElecQSTabTwinCut_VdwLJFSw_F.cpp
        kernel_ElecQSTabTwinCut_VdwLJFSw_VF.cpp
        kernel_ElecQSTabTwinCut_VdwLJFSw_VgrpF.cpp
        kernel_ElecQSTabTwinCut_VdwLJPSw_F.cpp
        kernel_ElecQSTabTwinCut_VdwLJPSw_VF.cpp
        kernel_ElecQSTabTwinCut_VdwLJPSw_VgrpF.cpp
        kernel_ElecQSTabTwinCut_VdwLJ_VF.cpp
        kernel_ElecQSTabTwinCut_VdwLJ_VgrpF.cpp
        kernel_ElecQSTab_VdwLJCombGeom_F.cpp
        kernel_ElecQSTab_VdwLJCombGeom_VF.cpp
        kernel_ElecQSTab_VdwLJCombGeom_VgrpF.cpp
        kernel_ElecQSTab_VdwLJCombLB_F.cpp
        kernel_ElecQSTab_VdwLJCombLB_VF.cpp
        kernel_ElecQSTab_VdwLJCombLB_VgrpF.cpp
        kernel_ElecQSTab_VdwLJEwCombGeom_F.cpp
        kernel_ElecQSTab_VdwLJEwCombGeom_VF.cpp
        kernel_ElecQSTab_VdwLJEwCombGeom_VgrpF.cpp
        kernel_ElecQSTab_VdwLJ_F.cpp
        kernel_ElecQSTab_VdwLJFSw_F.cpp
        kernel_ElecQSTab_VdwLJFSw_VF.cpp
        kernel_ElecQSTab_VdwLJFSw_VgrpF.cpp
        kernel_ElecQSTab_VdwLJPSw_F.cpp
        kernel_ElecQSTab_VdwLJPSw_VF.cpp
        kernel_ElecQSTab_VdwLJPSw_VgrpF.cpp
        kernel_ElecQSTab_VdwLJ_VF.cpp
        kernel_ElecQSTab_VdwLJ_VgrpF.cpp
        kernel_ElecRF_VdwLJCombGeom_F.cpp
        kernel_ElecRF_VdwLJCombGeom_VF.cpp
        kernel_ElecRF_VdwLJCombGeom_VgrpF.cpp
        kernel_ElecRF_VdwLJCombLB_F.cpp
        kernel_ElecRF_VdwLJCombLB_VF.cpp
        kernel_ElecRF_VdwLJCombLB_VgrpF.cpp
        kernel_ElecRF_VdwLJEwCombGeom_F.cpp
        kernel_ElecRF_VdwLJEwCombGeom_VF.cpp
        kernel_ElecRF_VdwLJEwCombGeom_VgrpF.cpp
        kernel_ElecRF_VdwLJ_F.cpp
        kernel_ElecRF_VdwLJFSw_F.cpp
        kernel_ElecRF_VdwLJFSw_VF.cpp
        kernel_ElecRF_VdwLJFSw_VgrpF.cpp
        kernel_ElecRF_VdwLJPSw_F.cpp
        kernel_ElecRF_VdwLJPSw_VF.cpp
        kernel_ElecRF_VdwLJPSw_VgrpF.cpp
        kernel_ElecRF_VdwLJ_VF.cpp
        kernel_ElecRF_VdwLJ_VgrpF.cpp
        )
endif()

set(LIBGROMACS_SOURCES ${LIBGROMACS_SOURCES} ${KERNEL_SOURCES} PARENT_SCOPE)
