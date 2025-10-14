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
 * \brief Declaration of class which receives coordinates to GPU memory on PME task
 *
 * \author Alan Gray <alang@nvidia.com>
 *
 * \ingroup module_ewald
 */
#ifndef GMX_PMECOORDINATERECEIVERGPU_IMPL_H
#define GMX_PMECOORDINATERECEIVERGPU_IMPL_H

#include <vector>

#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
#include "gromacs/utility/arrayref.h"

class GpuEventSynchronizer;

namespace gmx
{

/*! \internal
 *  \brief Object to manage communications with a specific PP rank
 */
struct PpCommManager
{
    //! Details of PP rank that may be updated after repartitioning
    const PpRanks& ppRank;
    //! Stream used communication with for PP rank
    std::unique_ptr<DeviceStream> stream;
    //! Synchronization event to receive from PP rank
    GpuEventSynchronizer* sync = nullptr;
    //! Synchronization event to indicate that all communications in \ref stream are complete.
    std::unique_ptr<GpuEventSynchronizer> ready;
    //! Range of atoms corresponding to PP rank
    std::tuple<int, int> atomRange = { 0, 0 };
};

/*! \internal \brief Class with interfaces and data for CUDA version of PME coordinate receiving functionality */
class PmeCoordinateReceiverGpu::Impl
{

public:
    /*! \brief Creates PME GPU coordinate receiver object
     * \param[in] comm            Communicator used for simulation
     * \param[in] deviceContext   GPU context
     * \param[in] ppRanks         List of PP ranks
     */
    Impl(MPI_Comm comm, const DeviceContext& deviceContext, gmx::ArrayRef<const PpRanks> ppRanks);
    ~Impl();

    /*! \brief
     * Re-initialize: set atom ranges and, for thread-MPI case,
     * send coordinates buffer address to PP rank.
     * This is required after repartitioning since atom ranges and
     * buffer allocations may have changed.
     * \param[in] d_x   coordinates buffer in GPU memory
     */
    void reinitCoordinateReceiver(DeviceBuffer<RVec> d_x);

    /*! \brief
     * Receive coordinate synchronizer pointer from the PP ranks.
     * \param[in] ppRank  PP rank to receive the synchronizer from.
     */
    void receiveCoordinatesSynchronizerFromPpPeerToPeer(int ppRank);

    /*! \brief
     * Used for lib MPI, receives co-ordinates from PP ranks
     * \param[in] recvbuf      coordinates buffer in GPU memory
     * \param[in] numAtoms     starting element in buffer
     * \param[in] numBytes     number of bytes to transfer
     * \param[in] ppRank       PP rank to send data
     * \param[in] senderIndex  Index of PP rank within those involved in communication with this PME rank
     */
    void launchReceiveCoordinatesFromPpGpuAwareMpi(DeviceBuffer<RVec> recvbuf,
                                                   int                numAtoms,
                                                   int                numBytes,
                                                   int                ppRank,
                                                   int                senderIndex);

    /*! \brief
     * Return PP co-ordinate transfer event received from PP
     * rank determined from pipeline stage, for consumer to enqueue
     * \param[in] pipelineStage  stage of pipeline corresponding to this transfer
     * \returns                  tuple with rank of sending PP task and corresponding event
     */
    std::tuple<int, GpuEventSynchronizer*> receivePpCoordinateSendEvent(int pipelineStage);

    /*! \brief
     * Wait for coordinates from any PP rank
     * \returns                  rank of sending PP task
     */
    int waitForCoordinatesFromAnyPpRank();

    /*! \brief
     * Return pointer to stream associated with specific PP rank sender index
     * \param[in] senderIndex    Index of sender PP rank.
     */
    DeviceStream* ppCommStream(int senderIndex);

    /*! \brief
     * Returns range of atoms involved in communication associated with specific PP rank sender
     * index \param[in] senderIndex    Index of sender PP rank.
     */
    std::tuple<int, int> ppCommAtomRange(int senderIndex);

    /*! \brief
     * Return number of PP ranks involved in PME-PP communication
     */
    int ppCommNumSenderRanks();

    /*! \brief
     * Mark an event in the sender stream \p senderIndex and enqueue it into \p stream.
     */
    void insertAsDependencyIntoStream(int senderIndex, const DeviceStream& stream);

private:
    //! communicator for simulation
    MPI_Comm comm_;
    //! MPI requests, one per PP rank
    std::vector<MPI_Request> requests_;
    //! GPU context handle (not used in CUDA)
    const DeviceContext& deviceContext_;
    //! Communication manager objects corresponding to multiple sending PP ranks
    std::vector<PpCommManager> ppCommManagers_;
};

} // namespace gmx

#endif
