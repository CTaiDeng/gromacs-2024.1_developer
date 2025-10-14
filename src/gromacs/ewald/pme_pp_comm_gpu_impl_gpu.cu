/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 *
 * \brief Implements backend-specific part of PME-PP communication using CUDA.
 *
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/typecasts.cuh"

#include "pme_pp_comm_gpu_impl.h"
#include "pme_pp_communication.h"

namespace gmx
{

void PmePpCommGpu::Impl::sendCoordinatesToPmePeerToPeer(Float3*               sendPtr,
                                                        int                   sendSize,
                                                        GpuEventSynchronizer* coordinatesReadyOnDeviceEvent)
{
    // ensure stream waits until coordinate data is available on device
    if (coordinatesReadyOnDeviceEvent)
    {
        coordinatesReadyOnDeviceEvent->enqueueWaitEvent(pmePpCommStream_);
    }

    cudaError_t stat = cudaMemcpyAsync(remotePmeXBuffer_,
                                       sendPtr,
                                       sendSize * DIM * sizeof(float),
                                       cudaMemcpyDefault,
                                       pmePpCommStream_.stream());
    CU_RET_ERR(stat, "cudaMemcpyAsync on Send to PME CUDA direct data transfer failed");

#if GMX_MPI
    // Record and send event to allow PME task to sync to above transfer before commencing force calculations
    pmeCoordinatesSynchronizer_.markEvent(pmePpCommStream_);
    GpuEventSynchronizer* pmeSync = &pmeCoordinatesSynchronizer_;
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    MPI_Send(&pmeSync, sizeof(GpuEventSynchronizer*), MPI_BYTE, pmeRank_, 0, comm_);
#endif
}

} // namespace gmx
