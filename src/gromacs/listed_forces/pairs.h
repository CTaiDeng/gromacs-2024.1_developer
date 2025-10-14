/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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
 * \brief This file declares functions for "pair" interactions
 * (i.e. listed non-bonded interactions, e.g. 1-4 interactions)
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inlibraryapi
 * \ingroup module_listed_forces
 */
#ifndef GMX_LISTED_FORCES_PAIRS_H
#define GMX_LISTED_FORCES_PAIRS_H

#include <vector>

#include "gromacs/math/vec.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/real.h"

struct gmx_grppairener_t;
struct t_forcerec;
struct t_pbc;

namespace gmx
{
class StepWorkload;
template<typename>
class ArrayRef;
} // namespace gmx

/*! \brief Calculate VdW/charge listed pair interactions (usually 1-4
 * interactions).
 *
 * global_atom_index is only passed for printing error messages.
 */
void do_pairs(int                                 ftype,
              int                                 nbonds,
              const t_iatom                       iatoms[],
              const t_iparams                     iparams[],
              const rvec                          x[],
              rvec4                               f[],
              rvec                                fshift[],
              const struct t_pbc*                 pbc,
              const real*                         lambda,
              real*                               dvdl,
              gmx::ArrayRef<const real>           chargeA,
              gmx::ArrayRef<const real>           chargeB,
              gmx::ArrayRef<const bool>           atomIsPerturbed,
              gmx::ArrayRef<const unsigned short> cENER,
              int                                 numEnergyGroups,
              const t_forcerec*                   fr,
              bool                                havePerturbedPairs,
              const gmx::StepWorkload&            stepWork,
              gmx_grppairener_t*                  grppener,
              int*                                global_atom_index);

#endif
