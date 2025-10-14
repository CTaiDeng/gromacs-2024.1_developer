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

#ifndef GMX_MDLIB_STAT_H
#define GMX_MDLIB_STAT_H

#include <cstdint>

#include "gromacs/math/vectypes.h"

class gmx_ekindata_t;
struct gmx_enerdata_t;
struct t_vcm;
struct t_inputrec;
struct t_commrec;

namespace gmx
{
template<typename T>
class ArrayRef;
class ObservablesReducer;
} // namespace gmx

typedef struct gmx_global_stat* gmx_global_stat_t;

gmx_global_stat_t global_stat_init(const t_inputrec* ir);

void global_stat_destroy(gmx_global_stat_t gs);

/*! \brief All-reduce energy-like quantities over cr->mpi_comm_mysim  */
void global_stat(const gmx_global_stat&   gs,
                 const t_commrec*         cr,
                 gmx_enerdata_t*          enerd,
                 tensor                   fvir,
                 tensor                   svir,
                 const t_inputrec&        inputrec,
                 gmx_ekindata_t*          ekind,
                 t_vcm*                   vcm,
                 gmx::ArrayRef<real>      sig,
                 bool                     bSumEkinhOld,
                 int                      flags,
                 int64_t                  step,
                 gmx::ObservablesReducer* observablesReducer);

/*! \brief Returns TRUE if io should be done */
inline bool do_per_step(int64_t step, int64_t nstep)
{
    if (nstep != 0)
    {
        return (step % nstep) == 0;
    }
    else
    {
        return false;
    }
}

#endif // GMX_MDLIB_STAT_H
