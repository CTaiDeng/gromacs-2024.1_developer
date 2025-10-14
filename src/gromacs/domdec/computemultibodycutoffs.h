/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 *
 * \brief This file declares the function for computing the required
 * cutoff distance for inter-domain multi-body interactions, when
 * those exist.
 *
 * \inlibraryapi
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_COMPUTEMULTIBODYCUTOFFS_H
#define GMX_DOMDEC_COMPUTEMULTIBODYCUTOFFS_H

#include "gromacs/math/vectypes.h"

struct gmx_mtop_t;
struct t_inputrec;

namespace gmx
{
template<typename>
class ArrayRef;
class MDLogger;
enum class DDBondedChecking : bool;
} // namespace gmx

/*! \brief Calculate the maximum distance involved in 2-body and multi-body bonded interactions */
void dd_bonded_cg_distance(const gmx::MDLogger&           mdlog,
                           const gmx_mtop_t&              mtop,
                           const t_inputrec&              ir,
                           gmx::ArrayRef<const gmx::RVec> x,
                           const matrix                   box,
                           gmx::DDBondedChecking          ddBondedChecking,
                           real*                          r_2b,
                           real*                          r_mb);

#endif
