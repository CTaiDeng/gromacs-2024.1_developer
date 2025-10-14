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
 * \brief Implememnts backend-specific code for PME-PP communication using CUDA.
 *
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "config.h"

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/utility/gmxmpi.h"

#include "pme_force_sender_gpu_impl.h"

namespace gmx
{

/*! \brief Send PME synchronizer directly using CUDA memory copy */
void PmeForceSenderGpu::Impl::sendFToPpPeerToPeer(int ppRank, int numAtoms, bool sendForcesDirectToPpGpu)
{

    GMX_ASSERT(GMX_THREAD_MPI, "sendFToPpCudaDirect is expected to be called only for Thread-MPI");

#if GMX_MPI
    Float3* pmeRemoteForcePtr = (sendForcesDirectToPpGpu || stageThreadMpiGpuCpuComm_)
                                        ? ppCommManagers_[ppRank].pmeRemoteGpuForcePtr
                                        : ppCommManagers_[ppRank].pmeRemoteCpuForcePtr;

    pmeForcesReady_->enqueueWaitEvent(*ppCommManagers_[ppRank].stream);

    // Push data to remote GPU's memory
    cudaError_t stat = cudaMemcpyAsync(asFloat3(pmeRemoteForcePtr),
                                       ppCommManagers_[ppRank].localForcePtr,
                                       numAtoms * sizeof(rvec),
                                       cudaMemcpyDefault,
                                       ppCommManagers_[ppRank].stream->stream());
    CU_RET_ERR(stat, "cudaMemcpyAsync on Recv from PME CUDA direct data transfer failed");

    if (stageThreadMpiGpuCpuComm_ && !sendForcesDirectToPpGpu)
    {
        // Perform local D2H (from remote GPU memory to remote PP rank's CPU memory)
        // to finalize staged data transfer
        stat = cudaMemcpyAsync(ppCommManagers_[ppRank].pmeRemoteCpuForcePtr,
                               ppCommManagers_[ppRank].pmeRemoteGpuForcePtr,
                               numAtoms * sizeof(rvec),
                               cudaMemcpyDefault,
                               ppCommManagers_[ppRank].stream->stream());
        CU_RET_ERR(stat, "cudaMemcpyAsync on local device to host transfer of PME forces failed");
    }

    ppCommManagers_[ppRank].event->markEvent(*ppCommManagers_[ppRank].stream);
    std::atomic<bool>* tmpPpCommEventRecordedPtr =
            reinterpret_cast<std::atomic<bool>*>(ppCommManagers_[ppRank].eventRecorded.get());
    tmpPpCommEventRecordedPtr->store(true, std::memory_order_release);
#else
    GMX_UNUSED_VALUE(ppRank);
    GMX_UNUSED_VALUE(numAtoms);
#endif
}

} // namespace gmx
