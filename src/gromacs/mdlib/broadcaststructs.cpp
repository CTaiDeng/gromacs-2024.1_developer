/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

/* This file is completely threadsafe - keep it that way! */
#include "gmxpre.h"

#include "broadcaststructs.h"

#include "gromacs/fileio/tpxio.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/utility/enumerationhelpers.h"

template<typename AllocatorType>
static void bcastPaddedRVecVector(MPI_Comm                                     communicator,
                                  gmx::PaddedVector<gmx::RVec, AllocatorType>* v,
                                  int                                          numAtoms)
{
    v->resizeWithPadding(numAtoms);
    nblock_bc(communicator, makeArrayRef(*v));
}

void broadcastStateWithoutDynamics(MPI_Comm communicator,
                                   bool     useDomainDecomposition,
                                   bool     isParallelRun,
                                   t_state* state)
{
    GMX_RELEASE_ASSERT(!useDomainDecomposition,
                       "broadcastStateWithoutDynamics should only be used for special cases "
                       "without domain decomposition");

    if (!isParallelRun)
    {
        return;
    }

    /* Broadcasts the state sizes and flags from the main to all ranks
     * in cr->mpi_comm_mygroup.
     */
    int numAtoms = state->numAtoms();
    block_bc(communicator, numAtoms);
    state->changeNumAtoms(numAtoms);
    int flags = state->flags();
    block_bc(communicator, flags);
    state->setFlags(flags);

    for (auto i : gmx::EnumerationArray<StateEntry, bool>::keys())
    {
        if (state->hasEntry(i))
        {
            switch (i)
            {
                case StateEntry::Lambda:
                    nblock_bc(communicator,
                              static_cast<int>(FreeEnergyPerturbationCouplingType::Count),
                              state->lambda.data());
                    break;
                case StateEntry::FepState: block_bc(communicator, state->fep_state); break;
                case StateEntry::Box: block_bc(communicator, state->box); break;
                case StateEntry::X:
                    bcastPaddedRVecVector(communicator, &state->x, state->numAtoms());
                    break;
                default:
                    GMX_RELEASE_ASSERT(false,
                                       "The state has a dynamic entry, while no dynamic entries "
                                       "should be present");
                    break;
            }
        }
    }
}

static void bc_tpxheader(MPI_Comm communicator, TpxFileHeader* tpx)
{
    block_bc(communicator, tpx->bIr);
    block_bc(communicator, tpx->bBox);
    block_bc(communicator, tpx->bTop);
    block_bc(communicator, tpx->bX);
    block_bc(communicator, tpx->bV);
    block_bc(communicator, tpx->bF);
    block_bc(communicator, tpx->natoms);
    block_bc(communicator, tpx->ngtc);
    block_bc(communicator, tpx->lambda);
    block_bc(communicator, tpx->fep_state);
    block_bc(communicator, tpx->sizeOfTprBody);
    block_bc(communicator, tpx->fileVersion);
    block_bc(communicator, tpx->fileGeneration);
    block_bc(communicator, tpx->isDouble);
}

static void bc_tprCharBuffer(MPI_Comm communicator, bool isMainRank, std::vector<char>* charBuffer)
{
    std::size_t elements = charBuffer->size();
    block_bc(communicator, elements);

    nblock_abc(isMainRank, communicator, elements, charBuffer);
}

void init_parallel(MPI_Comm                    communicator,
                   bool                        isMainRank,
                   t_inputrec*                 inputrec,
                   gmx_mtop_t*                 mtop,
                   PartialDeserializedTprFile* partialDeserializedTpr)
{
    bc_tpxheader(communicator, &partialDeserializedTpr->header);
    bc_tprCharBuffer(communicator, isMainRank, &partialDeserializedTpr->body);
    if (!isMainRank)
    {
        completeTprDeserialization(partialDeserializedTpr, inputrec, mtop);
    }
}
