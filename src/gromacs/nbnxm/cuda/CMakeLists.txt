# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2012- The GROMACS Authors
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

if(GMX_GPU_CUDA)
    if(NOT GMX_CUDA_NB_SINGLE_COMPILATION_UNIT)
        set(NBNXM_CUDA_KERNEL_SOURCES
                nbnxm_cuda_kernel_F_noprune.cu
                nbnxm_cuda_kernel_F_prune.cu
                nbnxm_cuda_kernel_VF_noprune.cu
                nbnxm_cuda_kernel_VF_prune.cu
                nbnxm_cuda_kernel_pruneonly.cu)
    endif()

    file(GLOB NBNXM_CUDA_SOURCES
         nbnxm_cuda.cu
         nbnxm_cuda_data_mgmt.cu
         nbnxm_cuda_jit_support.cu
         nbnxm_gpu_buffer_ops_internal.cu
         ${NBNXM_CUDA_KERNEL_SOURCES})
    set(NBNXM_SOURCES ${NBNXM_SOURCES} ${NBNXM_CUDA_SOURCES} PARENT_SCOPE)
endif()
