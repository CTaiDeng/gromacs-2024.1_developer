/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2005- The GROMACS Authors
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

/*! \internal \file
 *
 * \brief This file declares functions for domdec to use
 * while managing inter-atomic constraints.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_DOMDEC_CONSTRAINTS_H
#define GMX_DOMDEC_DOMDEC_CONSTRAINTS_H

#include <memory>

#include "gromacs/domdec/hashedmap.h"
#include "gromacs/utility/arrayref.h"

namespace gmx
{
class Constraints;
}

struct gmx_domdec_t;
struct gmx_mtop_t;
struct InteractionList;

/*! \brief Struct used during constraint setup with domain decomposition */
struct gmx_domdec_constraints_t
{
    //! @cond Doxygen_Suppress
    std::vector<int> molb_con_offset; /**< Offset in the constraint array for each molblock */
    std::vector<int> molb_ncon_mol; /**< The number of constraints per molecule for each molblock */

    int ncon; /**< The fully local and conneced constraints */
    /* The global constraint number, only required for clearing gc_req */
    std::vector<int> con_gl;     /**< Global constraint indices for local constraints */
    std::vector<int> con_nlocat; /**< Number of local atoms (2/1/0) for each constraint */

    std::vector<bool> gc_req; /**< Boolean that tells if a global constraint index has been requested; note: size global #constraints */

    /* Hash table for keeping track of requests */
    std::unique_ptr<gmx::HashedMap<int>> ga2la; /**< Global to local communicated constraint atom only index */

    /* Multi-threading stuff */
    int                          nthread; /**< Number of threads used for DD constraint setup */
    std::vector<InteractionList> ils;     /**< Constraint ilist working arrays, size \p nthread */

    /* Buffers for requesting atoms */
    std::vector<std::vector<int>> requestedGlobalAtomIndices; /**< Buffers for requesting global atom indices, one per thread */

    //! @endcond
};

/*! \brief Clears the local indices for the constraint communication setup */
void dd_clear_local_constraint_indices(gmx_domdec_t* dd);

/*! \brief Sets up communication and atom indices for all local+connected constraints */
int dd_make_local_constraints(struct gmx_domdec_t*           dd,
                              int                            at_start,
                              const struct gmx_mtop_t&       mtop,
                              gmx::ArrayRef<const int64_t>   atomInfo,
                              gmx::Constraints*              constr,
                              int                            nrec,
                              gmx::ArrayRef<InteractionList> il_local);

/*! \brief Initializes the data structures for constraint communication */
void init_domdec_constraints(gmx_domdec_t* dd, const gmx_mtop_t& mtop);

#endif
