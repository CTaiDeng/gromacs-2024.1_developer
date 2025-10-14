/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 * Declares the PairlistSet class
 *
 * There is one PairlistSet object per locality. A PairlistSet
 * holds a list of CPU- or GPU-type pairlist objects, one for each thread,
 * as well as helper objects to construct each of those pairlists.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_PAIRLISTSET_H
#define GMX_NBNXM_PAIRLISTSET_H

#include <memory>

#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

#include "pairlist.h"

struct nbnxn_atomdata_t;
struct PairlistParams;
struct PairsearchWork;
struct SearchCycleCounting;
struct t_nrnb;

namespace gmx
{
template<typename>
class ListOfLists;
}

namespace Nbnxm
{
class GridSet;
}

/*! \internal
 * \brief An object that holds the local or non-local pairlists
 */
class PairlistSet
{
public:
    //! Constructor: initializes the pairlist set as empty
    PairlistSet(const PairlistParams& listParams);

    ~PairlistSet();

    //! Constructs the pairlists in the set using the coordinates in \p nbat
    void constructPairlists(gmx::InteractionLocality      locality,
                            const Nbnxm::GridSet&         gridSet,
                            gmx::ArrayRef<PairsearchWork> searchWork,
                            nbnxn_atomdata_t*             nbat,
                            const gmx::ListOfLists<int>&  exclusions,
                            int                           minimumIlistCountForGpuBalancing,
                            t_nrnb*                       nrnb,
                            SearchCycleCounting*          searchCycleCounting);

    //! Dispatch the kernel for dynamic pairlist pruning
    void dispatchPruneKernel(const nbnxn_atomdata_t* nbat, gmx::ArrayRef<const gmx::RVec> shift_vec);

    //! Returns the lists of CPU pairlists
    gmx::ArrayRef<const NbnxnPairlistCpu> cpuLists() const { return cpuLists_; }

    //! Returns a pointer to the GPU pairlist, nullptr when not present
    const NbnxnPairlistGpu* gpuList() const
    {
        if (!gpuLists_.empty())
        {
            return &gpuLists_[0];
        }
        else
        {
            return nullptr;
        }
    }

    //! Returns the lists of free-energy pairlists, empty when nonbonded interactions are not perturbed
    gmx::ArrayRef<const std::unique_ptr<t_nblist>> fepLists() const { return fepLists_; }

    //! Returns the number of perturbed excluded pairs that are within distance rlist
    int numPerturbedExclusionsWithinRlist() const { return numPerturbedExclusionsWithinRlist_; }

private:
    //! List of pairlists in CPU layout
    std::vector<NbnxnPairlistCpu> cpuLists_;
    //! List of working list for rebalancing CPU lists
    std::vector<NbnxnPairlistCpu> cpuListsWork_;
    //! List of pairlists in GPU layout
    std::vector<NbnxnPairlistGpu> gpuLists_;
    //! Pairlist parameters describing setup and ranges
    const PairlistParams& params_;
    //! Tells whether multiple lists get merged into one (the first) after creation
    bool combineLists_;
    //! Tells whether the lists is of CPU type, otherwise GPU type
    gmx_bool isCpuType_;
    //! Lists for perturbed interactions in simple atom-atom layout
    std::vector<std::unique_ptr<t_nblist>> fepLists_;
    //! The number of excluded perturbed interaction within rlist
    int numPerturbedExclusionsWithinRlist_ = 0;

public:
    /* Pair counts for flop counting */
    //! Total number of atom pairs for LJ+Q kernel
    int natpair_ljq_;
    //! Total number of atom pairs for LJ kernel
    int natpair_lj_;
    //! Total number of atom pairs for Q kernel
    int natpair_q_;
};

#endif
