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

#ifndef GMX_MDLIB_TRAJECTORY_WRITING_H
#define GMX_MDLIB_TRAJECTORY_WRITING_H

#include <cstdio>

#include "gromacs/mdlib/mdoutf.h"

class gmx_ekindata_t;
struct gmx_mtop_t;
struct ObservablesHistory;
struct t_commrec;
struct t_filenm;
struct t_forcerec;

namespace gmx
{
class EnergyOutput;
}

//! The current state of ekindata as passed to do_md_trajectory_writing()
enum class EkindataState
{
    NotUsed,               //!< ekindata is not used this step and should thus not be checkpointed
    UsedNeedToReduce,      //!< ekindata is used this step and terms need to be reduced
    UsedDoNotNeedToReduce, //!< ekindata is used this step and no reduction is needed
};

/*! \brief Wrapper routine for writing trajectories during mdrun
 *
 * This routine does communication (e.g. collecting distributed coordinates).
 *
 * The kinetic energy data \p ekind is only used at steps where energies are
 * calculated or temperature or pressure coupling is done. Thus this data only
 * needs to be written to checkpoint at such steps. It might also contain
 * local contributions that are not yet reduced over ranks. The state of
 * \p ekind is described by \p ekindataState.
 */
void do_md_trajectory_writing(FILE*                          fplog,
                              struct t_commrec*              cr,
                              int                            nfile,
                              const t_filenm                 fnm[],
                              int64_t                        step,
                              int64_t                        step_rel,
                              double                         t,
                              const t_inputrec*              ir,
                              t_state*                       state,
                              t_state*                       state_global,
                              ObservablesHistory*            observablesHistory,
                              const gmx_mtop_t&              top_global,
                              t_forcerec*                    fr,
                              gmx_mdoutf_t                   outf,
                              const gmx::EnergyOutput&       energyOutput,
                              gmx_ekindata_t*                ekind,
                              gmx::ArrayRef<const gmx::RVec> f,
                              gmx_bool                       bCPT,
                              gmx_bool                       bRerunMD,
                              gmx_bool                       bLastStep,
                              gmx_bool                       bDoConfOut,
                              EkindataState                  ekindataState);

#endif
