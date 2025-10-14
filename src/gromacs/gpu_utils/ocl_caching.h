/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
 * Copyright (C) 2025 GaoZheng
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

/*! \internal \file
 *  \brief Declare infrastructure for managing caching of OpenCL
 *  JIT-ted binaries
 *
 *  This functionality is currently disabled in compileProgram()
 *
 *  \author Dimitrios Karkoulis <dimitris.karkoulis@gmail.com>
 *  \author Anca Hamuraru <anca@streamcomputing.eu>
 *  \author Teemu Virolainen <teemu@streamcomputing.eu>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 */
#ifndef GMX_GPU_UTILS_OCL_CACHING_H
#define GMX_GPU_UTILS_OCL_CACHING_H

#include <string>

#include "gromacs/gpu_utils/oclutils.h"

namespace gmx
{
namespace ocl
{

/*! \brief Construct the name for the binary cache file
 *
 * \param[in]  kernelFilename  Name of the kernel from which the binary will be compiled.
 * \param[in]  deviceId        ID of the device upon which the binary is used.
 *
 * \todo The set of preprocessor options should also form part of the
 * identification of the cached binary. Also perhaps compiler, runtime
 * and device version info?
 *
 * \todo Mutual exclusion of ranks and nodes should also be implemented
 * if/when caching is re-enabled.
 *
 * \returns The name of the cache file.
 */
std::string makeBinaryCacheFilename(const std::string& kernelFilename, cl_device_id deviceId);

/*! \brief Check if there's a valid cache available, and return it if so
 *
 * \param[in]  filename   Name of valid file containing the binary cache
 * \param[in]  context    The OpenCL context
 * \param[in]  deviceId   The ID of the device on which to use the program
 *
 * \returns The OpenCL program read from the cache
 *
 * \throws InternalError  if an OpenCL error was encountered
 *         FileIOError    if the file could not be opened
 */
cl_program makeProgramFromCache(const std::string& filename, cl_context context, cl_device_id deviceId);

/*! \brief Implement caching of OpenCL binaries
 *
 * \param[in] program     Index of program to cache
 * \param[in] filename    Name of file to use for the cache
 *
 * \throws InternalError  if an OpenCL error was encountered
 *         FileIOError    if the file could not be opened
 */
void writeBinaryToCache(cl_program program, const std::string& filename);

} // namespace ocl
} // namespace gmx

#endif
