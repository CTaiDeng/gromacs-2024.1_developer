/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

/*! \file
 * \internal
 *
 * \brief Declares constants and helper functions used when handling
 * bounding boxes for clusters of particles.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_BOUNDINGBOXES_H
#define GMX_NBNXM_BOUNDINGBOXES_H

#include "gromacs/simd/simd.h"

namespace Nbnxm
{

/*! \brief The number of bounds along one dimension of a bounding box */
static constexpr int c_numBoundingBoxBounds1D = 2;

} // namespace Nbnxm

#ifndef DOXYGEN

/*! \brief Bounding box calculations are (currently) always in single precision, so
 * we only need to check for single precision support here.
 * This uses less (cache-)memory and SIMD is faster, at least on x86.
 */
#    if GMX_SIMD4_HAVE_FLOAT
#        define NBNXN_SEARCH_BB_SIMD4 1
#    else
#        define NBNXN_SEARCH_BB_SIMD4 0
#    endif


#    if NBNXN_SEARCH_BB_SIMD4
/* Always use 4-wide SIMD for bounding box calculations */

#        if !GMX_DOUBLE
/* Single precision BBs + coordinates, we can also load coordinates with SIMD */
#            define NBNXN_SEARCH_SIMD4_FLOAT_X_BB 1
#        else
#            define NBNXN_SEARCH_SIMD4_FLOAT_X_BB 0
#        endif

/* Store bounding boxes corners as quadruplets: xxxxyyyyzzzz
 *
 * The packed bounding box coordinate stride is always set to 4.
 * With AVX we could use 8, but that turns out not to be faster.
 */
#        define NBNXN_BBXXXX 1

//! The number of bounding boxes in a pack, also the size of a pack along one dimension
static constexpr int c_packedBoundingBoxesDimSize = GMX_SIMD4_WIDTH;

//! Total number of corners (floats) in a pack of bounding boxes
static constexpr int c_packedBoundingBoxesSize =
        c_packedBoundingBoxesDimSize * DIM * Nbnxm::c_numBoundingBoxBounds1D;

//! Returns the starting index of the bounding box pack that contains the given cluster
static constexpr inline int packedBoundingBoxesIndex(int clusterIndex)
{
    return (clusterIndex / c_packedBoundingBoxesDimSize) * c_packedBoundingBoxesSize;
}

#    else /* NBNXN_SEARCH_BB_SIMD4 */

#        define NBNXN_SEARCH_SIMD4_FLOAT_X_BB 0
#        define NBNXN_BBXXXX 0

#    endif /* NBNXN_SEARCH_BB_SIMD4 */

#endif // !DOXYGEN

#endif
