/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
 * Copyright (C) 2025- GaoZheng
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * ---
 *
 * This file is part of a modified version of the GROMACS molecular simulation package.
 * For details on the original project, consult https://www.gromacs.org.
 *
 * To help fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

/*! \libinternal \file
 *  \brief Declare infrastructure for OpenCL JIT compilation
 *
 *  \author Dimitrios Karkoulis <dimitris.karkoulis@gmail.com>
 *  \author Anca Hamuraru <anca@streamcomputing.eu>
 *  \author Teemu Virolainen <teemu@streamcomputing.eu>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \inlibraryapi
 */
#ifndef GMX_GPU_UTILS_OCL_COMPILER_H
#define GMX_GPU_UTILS_OCL_COMPILER_H

#include <string>

#include "gromacs/gpu_utils/oclutils.h"
#include "gromacs/hardware/device_information.h"

namespace gmx
{
namespace ocl
{

/*! \brief Get the device-specific warp size
 *
 *  This is platform implementation dependent and seems to only work on the Nvidia and AMD
 * platforms! Nvidia reports 32, AMD for GPU 64. Intel seems to report 16, but that is not correct,
 *  as it execution width can be between 8-32 and it's picked per-kernel at compile-time.
 *  Therefore, for Intel it should actually be queried separately for each kernel (Issue #2520).
 *
 *  \param  context   Current OpenCL context
 *  \param  deviceId OpenCL device with the context
 *  \return cl_int value of the warp size
 *
 * \throws InternalError if an OpenCL error was encountered
 */
size_t getDeviceWarpSize(cl_context context, cl_device_id deviceId);


/*! \brief Get the kernel-specific warp size
 *
 *  \param  kernel   THe OpenCL kernel object
 *  \param  deviceId OpenCL device for which the kernel warp size is queried
 *  \return cl_int value of the warp size
 *
 * \throws InternalError if an OpenCL error was encountered
 */
size_t getKernelWarpSize(cl_kernel kernel, cl_device_id deviceId);

/*! \brief Compile the specified kernel for the context and device.
 *
 * \param[out] fplog                 Open file pointer for log output
 * \param[in]  kernelRelativePath    Relative path to the kernel in the source tree,
 *                                   e.g. "src/gromacs/mdlib/nbnxn_ocl" for NB kernels.
 * \param[in]  kernelBaseFilename    The name of the kernel source file to compile, e.g.
 *                                   "nbnxn_ocl_kernels.cl"
 * \param[in]  extraDefines          Preprocessor defines required by the calling code,
 *                                   e.g. for configuring the kernels
 * \param[in]  context               OpenCL context on the device to compile for
 * \param[in]  deviceId              OpenCL device id of the device to compile for
 * \param[in]  deviceVendor          Enumerator of the device vendor to compile for
 *
 * \returns The compiled OpenCL program
 *
 * \todo Consider whether we can parallelize the compilation of all
 * the kernels by compiling them in separate programs - but since the
 * resulting programs can't refer to each other, that might lead to
 * bloat of util code?
 *
 * \throws std::bad_alloc  if out of memory.
 *         FileIOError     if a file I/O error prevents returning a valid compiled program.
 *         InternalError   if an OpenCL API error prevents returning a valid compiled program. */
cl_program compileProgram(FILE*              fplog,
                          const std::string& kernelRelativePath,
                          const std::string& kernelBaseFilename,
                          const std::string& extraDefines,
                          cl_context         context,
                          cl_device_id       deviceId,
                          DeviceVendor       deviceVendor);

} // namespace ocl
} // namespace gmx

#endif
