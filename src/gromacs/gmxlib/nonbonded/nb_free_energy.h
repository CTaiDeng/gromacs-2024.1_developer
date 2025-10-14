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

#ifndef GMX_GMXLIB_NONBONDED_NB_FREE_ENERGY_H
#define GMX_GMXLIB_NONBONDED_NB_FREE_ENERGY_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"

struct t_forcerec;
struct t_nrnb;
struct t_nblist;
struct interaction_const_t;
namespace gmx
{
template<typename>
class ArrayRef;
template<typename>
class ArrayRefWithPadding;
} // namespace gmx

/*! \brief The non-bonded free-energy kernel
 *
 * Note that this uses a regular atom pair, not cluster pair, list.
 *
 * \throws InvalidInputError when an excluded pair is beyond the rcoulomb with reaction-field.
 */
void gmx_nb_free_energy_kernel(const t_nblist&                                  nlist,
                               const gmx::ArrayRefWithPadding<const gmx::RVec>& coords,
                               bool                                             useSimd,
                               int                                              ntype,
                               const interaction_const_t&                       ic,
                               gmx::ArrayRef<const gmx::RVec>                   shiftvec,
                               gmx::ArrayRef<const real>                        nbfp,
                               gmx::ArrayRef<const real>                        nbfp_grid,
                               gmx::ArrayRef<const real>                        chargeA,
                               gmx::ArrayRef<const real>                        chargeB,
                               gmx::ArrayRef<const int>                         typeA,
                               gmx::ArrayRef<const int>                         typeB,
                               int                                              flags,
                               gmx::ArrayRef<const real>                        lambda,
                               t_nrnb* gmx_restrict                             nrnb,
                               gmx::ArrayRefWithPadding<gmx::RVec>              threadForceBuffer,
                               rvec*               threadForceShiftBuffer,
                               gmx::ArrayRef<real> threadVc,
                               gmx::ArrayRef<real> threadVv,
                               gmx::ArrayRef<real> threadDvdl);

#endif
