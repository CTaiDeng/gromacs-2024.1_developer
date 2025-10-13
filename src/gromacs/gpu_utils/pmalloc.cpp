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

/*! \internal \file
 *  \brief Define functions for host-side memory handling when using OpenCL devices or no GPU device.
 *
 *  \author Anca Hamuraru <anca@streamcomputing.eu>
 */

#include "gmxpre.h"

#include "pmalloc.h"

#include "gromacs/utility/smalloc.h"

/*! \brief Allocates nbytes of host memory. Use pfree to free memory allocated with this function.
 *
 *  \todo
 *  This function should allocate page-locked memory to help reduce D2H and H2D
 *  transfer times, similar with pmalloc from pmalloc.cu.
 *
 * \param[in,out]    h_ptr   Pointer where to store the address of the newly allocated buffer.
 * \param[in]        nbytes  Size in bytes of the buffer to be allocated.
 */
void pmalloc(void** h_ptr, size_t nbytes, const DeviceContext* /*context*/)
{
    /* Need a temporary type whose size is 1 byte, so that the
     * implementation of snew_aligned can cope without issuing
     * warnings. */
    char** temporary = reinterpret_cast<char**>(h_ptr);

    /* 16-byte alignment is required by the neighbour-searching code,
     * because it uses four-wide SIMD for bounding-box calculation.
     * However, when we organize using page-locked memory for
     * device-host transfers, it will probably need to be aligned to a
     * 4kb page, like CUDA does. */
    snew_aligned(*temporary, nbytes, 16);
}

/*! \brief Frees memory allocated with pmalloc.
 *
 * \param[in]    h_ptr   Buffer allocated with pmalloc that needs to be freed.
 */
void pfree(void* h_ptr, const DeviceContext* /*context*/)
{

    if (h_ptr)
    {
        sfree_aligned(h_ptr);
    }
}

void pmallocSetDefaultDeviceContext(const DeviceContext* /*context*/)
{
    // We don't need context because we don't do anything device-specific.
}

void pmallocClearDefaultDeviceContext()
{
    // We don't need context because we don't do anything device-specific.
}
