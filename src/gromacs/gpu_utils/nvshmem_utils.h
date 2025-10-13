/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2023- The GROMACS Authors
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
 *
 * \brief Declarations for NVSHMEM initialization/finalize class.
 * gmxNvshmemHandle takes the MPI communicator and initializes the
 * NVSHMEM over all the ranks involved in the given MPI communicator.
 * This is a collective call for all the ranks in the given MPI comm.
 * After NVSHMEM initialization all NVSHMEM APIs can be safely used.
 *
 * \author Mahesh Doijade <mdoijade@nvidia.com>
 *
 * \ingroup module_gpu_utils
 * \inlibraryapi
 */

#ifndef GMX_NVSHMEM_UTILS_H_
#define GMX_NVSHMEM_UTILS_H_

#include "gromacs/utility/gmxmpi.h"

class gmxNvshmemHandle
{

private:
    MPI_Comm nvshmem_mpi_comm_;

public:
    gmxNvshmemHandle(MPI_Comm comm);

    ~gmxNvshmemHandle();
};

#endif
