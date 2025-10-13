/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 * \brief This file makes declarations used for building
 * the local topology
 *
 * \inlibraryapi
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_LOCALTOPOLOGY_H
#define GMX_DOMDEC_LOCALTOPOLOGY_H

#include "gromacs/math/vectypes.h"

struct gmx_domdec_t;
struct gmx_domdec_zones_t;
struct gmx_localtop_t;
struct gmx_mtop_t;
struct t_forcerec;
struct t_mdatoms;

namespace gmx
{
template<typename>
class ArrayRef;
}

/*! \brief Generate the local topology and virtual site data
 *
 * \returns Total count of bonded interactions in the local topology on this domain */
int dd_make_local_top(const gmx_domdec_t&            dd,
                      const gmx_domdec_zones_t&      zones,
                      int                            npbcdim,
                      matrix                         box,
                      rvec                           cellsize_min,
                      const ivec                     npulse,
                      t_forcerec*                    fr,
                      gmx::ArrayRef<const gmx::RVec> coordinates,
                      const gmx_mtop_t&              top,
                      gmx::ArrayRef<const int64_t>   atomInfo,
                      gmx_localtop_t*                ltop);

#endif
