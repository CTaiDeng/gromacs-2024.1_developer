# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2017- The GROMACS Authors
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

gmx_add_unit_test(GpuUtilsUnitTests gpu_utils-test HARDWARE_DETECTION
    CPP_SOURCE_FILES
        # Tests of code
        clfftinitializer.cpp
        device_availability.cpp
        device_stream_manager.cpp
        hostallocator.cpp
        pinnedmemorychecker.cpp
        
    GPU_CPP_SOURCE_FILES
        device_buffer.cpp
        gpueventsynchronizer.cpp
        typecasts.cpp

    CUDA_CU_SOURCE_FILES
        devicetransfers.cu
        typecasts_runner.cu

    OPENCL_CPP_SOURCE_FILES
        devicetransfers_ocl.cpp

    SYCL_CPP_SOURCE_FILES
        devicetransfers_sycl.cpp

    NON_GPU_CPP_SOURCE_FILES
        devicetransfers.cpp
    )
target_link_libraries(gpu_utils-test PRIVATE gpu_utils)
target_include_directories(gpu_utils-test PRIVATE ${PROJECT_SOURCE_DIR}/src/gromacs/math/tests)

if (GMX_NVSHMEM)
    gmx_add_mpi_unit_test(NvshmemTests nvshmem-test 2 HARDWARE_DETECTION NVSHMEM
    CUDA_CU_SOURCE_FILES
        nvshmem_simple_put.cu
    GPU_CPP_SOURCE_FILES
        nvshmem_test.cpp)

    target_include_directories(nvshmem-test PRIVATE ${PROJECT_SOURCE_DIR}/src/gromacs/math/tests)
    target_link_libraries(nvshmem-test PRIVATE gpu_utils)
endif()
