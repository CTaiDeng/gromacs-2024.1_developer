/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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

#ifndef GMX_MDLIB_EXPANDED_H
#define GMX_MDLIB_EXPANDED_H

#include <cstdio>

#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/coupling.h"

struct df_history_t;
class gmx_ekindata_t;
struct gmx_enerdata_t;
struct t_expanded;
struct t_extmass;
struct t_inputrec;
struct t_lambda;
struct t_simtemp;
class t_state;

namespace gmx
{
template<typename>
class ArrayRef;
}

void init_expanded_ensemble(bool bStateFromCP, const t_inputrec* ir, df_history_t* dfhist);

int ExpandedEnsembleDynamics(FILE*                               log,
                             const t_inputrec&                   ir,
                             const gmx_enerdata_t&               enerd,
                             gmx_ekindata_t*                     ekind,
                             t_state*                            state,
                             t_extmass*                          MassQ,
                             int                                 fep_state,
                             df_history_t*                       dfhist,
                             int64_t                             step,
                             rvec*                               v,
                             int                                 homenr,
                             gmx::ArrayRef<const unsigned short> cTC);

/*!
 * \brief Return a new lambda state using expanded ensemble
 *
 * \param log  File pointer to the log file
 * \param ir  The input record
 * \param enerd  Data for energy output.
 * \param fep_state  The current lambda state
 * \param dfhist  Free energy sampling history struct
 * \param step  The current simulation step
 * \return  The new lambda state
 */
int expandedEnsembleUpdateLambdaState(FILE*                 log,
                                      const t_inputrec*     ir,
                                      const gmx_enerdata_t* enerd,
                                      int                   fep_state,
                                      df_history_t*         dfhist,
                                      int64_t               step);

void PrintFreeEnergyInfoToFile(FILE*               outfile,
                               const t_lambda*     fep,
                               const t_expanded*   expand,
                               const t_simtemp*    simtemp,
                               const df_history_t* dfhist,
                               int                 fep_state,
                               int                 frequency,
                               int64_t             step);

#endif
