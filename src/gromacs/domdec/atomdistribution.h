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
 * \brief Declares the AtomDistribution struct.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */
#ifndef GMX_DOMDEC_ATOMDISTRIBUTION_H
#define GMX_DOMDEC_ATOMDISTRIBUTION_H

#include <array>
#include <limits>
#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"

namespace gmx
{

template<typename>
class ArrayRef;

} // namespace gmx

/*! \internal
 * \brief Distribution of atom groups over the domain (only available on the main rank)
 */
struct AtomDistribution
{
    /*! \internal
     * \brief Collection of local group and atom counts for a domain
     */
    struct DomainAtomGroups
    {
        gmx::ArrayRef<const int> atomGroups; /**< List of our atom groups */
        int                      numAtoms;   /**< Our number of local atoms */
    };

    /*! \brief Constructor */
    AtomDistribution(const ivec numCells, int numAtomGroups, int numAtoms);

    std::vector<DomainAtomGroups> domainGroups; /**< Group and atom division over ranks/domains */
    std::vector<int>              atomGroups; /**< The atom group division of the whole system, pointed into by counts[].atomGroups */

    /* Temporary buffers, stored permanently here to avoid reallocation */
    std::array<std::vector<real>, DIM> cellSizesBuffer; /**< Cell boundaries, sizes: num_cells_in_dim + 1 */
    std::vector<int>       intBuffer;  /**< Buffer for communicating cg and atom counts */
    std::vector<gmx::RVec> rvecBuffer; /**< Buffer for state scattering and gathering */
};

/*! \brief Returns state scatter/gather buffer atom counts and displacements
 *
 * NOTE: Should only be called with a pointer to a valid ma struct
 *       (only available on the main rank).
 */
void get_commbuffer_counts(AtomDistribution*         ma,
                           gmx::ArrayRef<const int>* counts,
                           gmx::ArrayRef<const int>* displacements);

#endif
