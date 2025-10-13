/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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
 * \brief
 * Declares the checkpoint handler class.
 *
 * This class sets the signal to checkpoint based on the elapsed simulation time,
 * and handles the signal if it is received. When handling the signal (via
 * decideIfCheckpointingThisStep()), it is deciding whether a checkpoint should be saved
 * at the current step. This can be due to a received signal, or if the current simulation
 * step is the last. This information can be queried via the isCheckpointingStep() function.
 *
 * The setting and handling is implemented in private functions. They are only called
 * if a respective boolean is true. For the trivial case of no checkpointing (or no checkpoint
 * signal setting on any other rank than main), the translation unit of the calling
 * function is therefore never left. In the future, this will be achieved by adding
 * (or not adding) handlers / setters to the task graph.
 *
 * \author Pascal Merz <pascal.merz@colorado.edu>
 * \inlibraryapi
 * \ingroup module_mdlib
 */
#ifndef GMX_MDLIB_CHECKPOINTHANDLER_H
#define GMX_MDLIB_CHECKPOINTHANDLER_H

#include <memory>

#include "gromacs/compat/pointers.h"
#include "gromacs/mdlib/simulationsignal.h"

struct gmx_walltime_accounting;

namespace gmx
{
/*! \brief Checkpoint signals
 *
 * Signals set and read by CheckpointHandler. Possible signals include
 *   * nothing to signal
 *   * do checkpoint (at next NS step)
 */
enum class CheckpointSignal
{
    noSignal     = 0,
    doCheckpoint = 1
};

/*! \libinternal
 * \brief Class handling the checkpoint signal
 *
 * Main rank sets the checkpointing signal periodically
 * All ranks receive checkpointing signal and set the respective flag
 */
class CheckpointHandler final
{
public:
    /*! \brief CheckpointHandler constructor
     *
     * Needs a non-null pointer to the signal which is reduced by compute_globals, and
     * (const) references to data it needs to determine whether a signal needs to be
     * set or handled.
     */
    CheckpointHandler(compat::not_null<SimulationSignal*> signal,
                      bool                                simulationsShareState,
                      bool                                neverUpdateNeighborList,
                      bool                                isMain,
                      bool                                writeFinalCheckpoint,
                      real                                checkpointingPeriod);

    /*! \brief Decides whether a checkpointing signal needs to be set
     *
     * Checkpointing signal is set based on the elapsed run time and the checkpointing
     * interval.
     */
    void setSignal(gmx_walltime_accounting* walltime_accounting) const
    {
        if (rankCanSetSignal_)
        {
            setSignalImpl(walltime_accounting);
        }
    }

    /*! \brief Decides whether a checkpoint shall be written at this step
     *
     * Checkpointing is done if this is not the initial step, and
     *   * a signal has been set and the current step is a neighborlist creation
     *     step, or
     *   * the current step is the last step and a the simulation is writing
     *     configurations.
     *
     * \todo Change these bools to enums to make calls more self-explanatory
     */
    void decideIfCheckpointingThisStep(bool bNS, bool bFirstStep, bool bLastStep)
    {
        if (checkpointingIsActive_)
        {
            decideIfCheckpointingThisStepImpl(bNS, bFirstStep, bLastStep);
        }
    }

    //! Query decision in decideIfCheckpointingThisStep()
    bool isCheckpointingStep() const { return checkpointThisStep_; }

private:
    void setSignalImpl(gmx_walltime_accounting* walltime_accounting) const;

    void decideIfCheckpointingThisStepImpl(bool bNS, bool bFirstStep, bool bLastStep);

    SimulationSignal& signal_;
    bool              checkpointThisStep_;
    int               numberOfNextCheckpoint_;

    const bool rankCanSetSignal_;
    const bool checkpointingIsActive_;
    const bool writeFinalCheckpoint_;
    const bool neverUpdateNeighborlist_;
    const real checkpointingPeriod_;
};
} // namespace gmx

#endif // GMX_MDLIB_CHECKPOINTHANDLER_H
