/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/*! \libinternal \file
 * \brief Declares functions to collect state data to the main rank.
 *
 * \author Berk Hess <hess@kth.se>
 * \inlibraryapi
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_COLLECT_H
#define GMX_DOMDEC_COLLECT_H

#include "gromacs/math/vectypes.h"

namespace gmx
{
template<typename>
class ArrayRef;
}
struct gmx_domdec_t;
class t_state;

/*! \brief Gathers rvec arrays \p localVector to \p globalVector on the main rank */
void dd_collect_vec(gmx_domdec_t*                  dd,
                    int                            ddpCount,
                    int                            ddpCountCgGl,
                    gmx::ArrayRef<const int>       localCGNumbers,
                    gmx::ArrayRef<const gmx::RVec> localVector,
                    gmx::ArrayRef<gmx::RVec>       globalVector);

/*! \brief Gathers state \p localState to \p globalState on the main rank */
void dd_collect_state(gmx_domdec_t* dd, const t_state* localState, t_state* globalState);

#endif
