/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/*! \internal \file
 *
 * \brief This file declares functions used by the domdec module
 * for (bounding) box and pbc information generation.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_BOX_H
#define GMX_DOMDEC_BOX_H

#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/gmxmpi.h"

namespace gmx
{
template<typename>
class ArrayRef;
}
struct gmx_ddbox_t;
struct gmx_domdec_t;
struct t_commrec;
struct t_inputrec;
enum class DDRole;

/*! \brief Set the box and PBC data in \p ddbox */
void set_ddbox(const gmx_domdec_t&            dd,
               bool                           mainRankHasTheSystemState,
               const matrix                   box,
               bool                           calculateUnboundedSize,
               gmx::ArrayRef<const gmx::RVec> x,
               gmx_ddbox_t*                   ddbox);

/*! \brief Set the box and PBC data in \p ddbox */
void set_ddbox_cr(DDRole                         ddRole,
                  MPI_Comm                       communicator,
                  const ivec*                    dd_nc,
                  const t_inputrec&              ir,
                  const matrix                   box,
                  gmx::ArrayRef<const gmx::RVec> x,
                  gmx_ddbox_t*                   ddbox);

/*! \brief Computes and returns a domain decomposition box */
gmx_ddbox_t get_ddbox(const ivec&                    numDomains,
                      const t_inputrec&              ir,
                      const matrix                   box,
                      gmx::ArrayRef<const gmx::RVec> x);

#endif
