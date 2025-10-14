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

/*! \libinternal \file
 *
 * \brief This file contains declarations necessary for low-level
 * functions for computing energies and forces for position
 * restraints.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inlibraryapi
 * \ingroup module_listed_forces
 */

#ifndef GMX_LISTED_FORCES_POSITION_RESTRAINTS_H
#define GMX_LISTED_FORCES_POSITION_RESTRAINTS_H

#include <cstdio>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

struct gmx_enerdata_t;
struct gmx_wallcycle;
struct t_forcerec;
class InteractionDefinitions;
struct t_nrnb;
struct t_pbc;

namespace gmx
{
class ForceWithVirial;
}

/*! \brief Helper function that wraps calls to posres */
void posres_wrapper(t_nrnb*                       nrnb,
                    const InteractionDefinitions& idef,
                    const struct t_pbc*           pbc,
                    const rvec*                   x,
                    gmx_enerdata_t*               enerd,
                    gmx::ArrayRef<const real>     lambda,
                    const t_forcerec*             fr,
                    gmx::ForceWithVirial*         forceWithVirial);

/*! \brief Helper function that wraps calls to posres for free-energy
    pertubation */
void posres_wrapper_lambda(struct gmx_wallcycle*         wcycle,
                           const InteractionDefinitions& idef,
                           const struct t_pbc*           pbc,
                           const rvec                    x[],
                           gmx_enerdata_t*               enerd,
                           gmx::ArrayRef<const real>     lambda,
                           const t_forcerec*             fr);

/*! \brief Helper function that wraps calls to fbposres for
    free-energy perturbation */
void fbposres_wrapper(t_nrnb*                       nrnb,
                      const InteractionDefinitions& idef,
                      const struct t_pbc*           pbc,
                      const rvec*                   x,
                      gmx_enerdata_t*               enerd,
                      const t_forcerec*             fr,
                      gmx::ForceWithVirial*         forceWithVirial);

#endif
