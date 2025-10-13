/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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
 * \brief Declares the routines for replica exchange.
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_mdrun
 */
#ifndef GMX_MDRUN_REPLICAEXCHANGE_H
#define GMX_MDRUN_REPLICAEXCHANGE_H

#include <cstdio>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct gmx_enerdata_t;
struct gmx_multisim_t;
struct t_commrec;
struct t_inputrec;
class t_state;

/*! \libinternal
 * \brief The parameters for the replica exchange algorithm. */
struct ReplicaExchangeParameters
{
    //! Interval in steps at which to attempt exchanges, 0 means no replica exchange.
    int exchangeInterval = 0;
    //! The number of exchanges to attempt at an exchange step.
    int numExchanges = 0;
    //! The random seed, -1 means generate a seed.
    int randomSeed = -1;
};

//! Abstract type for replica exchange
typedef struct gmx_repl_ex* gmx_repl_ex_t;

/*! \brief Setup function.
 *
 * Should only be called on the main ranks */
gmx_repl_ex_t init_replica_exchange(FILE*                            fplog,
                                    const gmx_multisim_t*            ms,
                                    int                              numAtomsInSystem,
                                    const t_inputrec*                ir,
                                    const ReplicaExchangeParameters& replExParams);

/*! \brief Attempts replica exchange.
 *
 * Should be called on all ranks.  When running each replica in
 * parallel, this routine collects the state on the main rank before
 * exchange.  With domain decomposition, the global state after
 * exchange is stored in state and still needs to be redistributed
 * over the ranks.
 *
 * \returns TRUE if the state has been exchanged.
 */
gmx_bool replica_exchange(FILE*                 fplog,
                          const t_commrec*      cr,
                          const gmx_multisim_t* ms,
                          gmx_repl_ex_t         re,
                          t_state*              state,
                          const gmx_enerdata_t* enerd,
                          t_state*              state_local,
                          int64_t               step,
                          real                  time);

/*! \brief Prints replica exchange statistics to the log file.
 *
 * Should only be called on the main ranks */
void print_replica_exchange_statistics(FILE* fplog, gmx_repl_ex_t re);

#endif
