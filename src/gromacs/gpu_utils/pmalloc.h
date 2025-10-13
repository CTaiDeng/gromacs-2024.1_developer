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
 *  \brief Declare functions for host-side memory handling.
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 *  \inlibraryapi
 */
#ifndef GMX_GPU_UTILS_PMALLOC_H
#define GMX_GPU_UTILS_PMALLOC_H

#include <cstdlib>

#include "gromacs/utility/basedefinitions.h"

class DeviceContext;

///@cond INTERNAL

/*! \brief Allocates nbytes of page-locked memory. */
void pmalloc(void** h_ptr, size_t nbytes, const DeviceContext* deviceContext = nullptr);

/*! \brief Frees page locked memory allocated with pmalloc. */
void pfree(void* h_ptr, const DeviceContext* deviceContext = nullptr);

//! \brief Set default device context for when it is not explicitly provided to \ref pmalloc or \ref pfree.
//
// Per thread.
// If a context was set before, all the memory allocated in it should have been freed before setting
// a new one.
//
// Only does something in the SYCL build, but for ease of maintenance please use in other builds as well.
void pmallocSetDefaultDeviceContext(const DeviceContext* deviceContext);

//! \brief Unset the default device context.
//
// Per thread.
// If a new context is being set by \ref pmallocSetDefaultDeviceContext, there is no need to call
// this function explicitly.
// If the default context for a thread was ever set, this function should be called before the end
// of the program to ensure proper resource release.
//
// Only does something in the SYCL build, but for ease of maintenance please use in other builds as well.
void pmallocClearDefaultDeviceContext();

///@endcond

#endif
