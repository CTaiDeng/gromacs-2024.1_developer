/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * \brief
 * Declares the GridSetData struct which holds grid data that is shared over all grids
 *
 * Also declares a struct for work data that is shared over grids.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_GRIDSETDATA_H
#define GMX_NBNXM_GRIDSETDATA_H

#include <vector>

#include "gromacs/gpu_utils/hostallocator.h"

namespace Nbnxm
{

/*! \internal
 * \brief Struct that holds grid data that is shared over all grids
 *
 * To enable a single coordinate and force array, a single cell range
 * is needed which covers all grids.
 */
struct GridSetData
{
    //! The cell indices for all atoms
    gmx::HostVector<int> cells;
    //! The atom indices for all atoms stored in cell order
    gmx::HostVector<int> atomIndices;
};

/*! \internal
 * \brief Working arrays for constructing a grid
 */
struct GridWork
{
    //! Number of atoms for each grid column
    std::vector<int> numAtomsPerColumn;
    //! Buffer for sorting integers
    std::vector<int> sortBuffer;
};

} // namespace Nbnxm

#endif
