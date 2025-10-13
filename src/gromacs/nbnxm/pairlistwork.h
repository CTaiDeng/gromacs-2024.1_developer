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

/*! \internal \file
 *
 * \brief Declares working data structures for the CPU and GPU pairlists
 *
 * \author Berk Hess <hess@kth.se>
 *
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_PAIRLISTWORK_H
#define GMX_NBNXM_PAIRLISTWORK_H

#include <memory>
#include <vector>

#include "grid.h"
#include "pairlist.h"

//! Working data for the actual i-supercell during pair search \internal
struct NbnxnPairlistCpuWork
{
    //! Struct for storing coordinates and bounding box for an i-entry during search \internal
    struct IClusterData
    {
        IClusterData() :
            bb(1),
            x(c_nbnxnCpuIClusterSize * DIM),
            xSimd(c_nbnxnCpuIClusterSize * DIM * GMX_REAL_MAX_SIMD_WIDTH)
        {
        }

        //! The bounding boxes, pbc shifted, for each cluster
        AlignedVector<Nbnxm::BoundingBox> bb;
        //! The coordinates, pbc shifted, for each atom
        std::vector<real> x;
        //! Aligned list for storing 4*DIM*GMX_SIMD_REAL_WIDTH reals
        AlignedVector<real> xSimd;
    };

    //! Protect data from cache pollution between threads
    gmx_cache_protect_t cp0;

    //! Work data for generating an IEntry in the pairlist
    IClusterData iClusterData;
    //! Temporary j-cluster list, used for sorting on exclusions
    std::vector<nbnxn_cj_t> cj;

    //! Nr. of cluster pairs without Coulomb for flop counting
    int ncj_noq;
    //! Nr. of cluster pairs with 1/2 LJ for flop count
    int ncj_hlj;

    //! Protect data from cache pollution between threads
    gmx_cache_protect_t cp1;
};

/* Working data for the actual i-supercell during pair search */
struct NbnxnPairlistGpuWork
{
    struct ISuperClusterData
    {
        ISuperClusterData();

        //! The bounding boxes, pbc shifted, for each cluster
        AlignedVector<Nbnxm::BoundingBox> bb;
        //! As bb, but in packed xxxx format
        AlignedVector<float> bbPacked;
        //! The coordinates, pbc shifted, for each atom
        AlignedVector<real> x;
        //! Aligned coordinate list used for 4*DIM*GMX_SIMD_REAL_WIDTH floats
        AlignedVector<real> xSimd;
    };

    NbnxnPairlistGpuWork();

    //! Protect data from cache pollution between threads
    gmx_cache_protect_t cp0;

    //! Work data for generating an i-entry in the pairlist
    ISuperClusterData iSuperClusterData;
    //! The current j-cluster index for the current list
    int cj_ind;
    //! Bounding box distance work array
    AlignedVector<float> distanceBuffer;

    //! Buffer for sorting list entries
    std::vector<int> sortBuffer;

    //! Second sci array, for sorting
    gmx::HostVector<nbnxn_sci_t> sci_sort;

    //! Protect data from cache pollution between threads
    gmx_cache_protect_t cp1;
};

#endif
