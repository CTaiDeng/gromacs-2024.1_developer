/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 * \brief
 * Declares the PairlistSets class
 *
 * This class holds the local and non-local pairlist sets.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_PAIRLISTSETS_H
#define GMX_NBNXM_PAIRLISTSETS_H

#include <memory>

#include "gromacs/mdtypes/locality.h"

#include "pairlistparams.h"

struct nbnxn_atomdata_t;
class PairlistSet;
enum class PairlistType;
class PairSearch;
struct t_nrnb;

namespace gmx
{
template<typename>
class ListOfLists;
}

//! Contains sets of pairlists \internal
class PairlistSets
{
public:
    //! Constructor
    PairlistSets(const PairlistParams& pairlistParams,
                 bool                  haveMultipleDomains,
                 int                   minimumIlistCountForGpuBalancing);

    //! Construct the pairlist set for the given locality
    void construct(gmx::InteractionLocality     iLocality,
                   PairSearch*                  pairSearch,
                   nbnxn_atomdata_t*            nbat,
                   const gmx::ListOfLists<int>& exclusions,
                   int64_t                      step,
                   t_nrnb*                      nrnb);

    //! Dispatches the dynamic pruning kernel for the given locality
    void dispatchPruneKernel(gmx::InteractionLocality       iLocality,
                             const nbnxn_atomdata_t*        nbat,
                             gmx::ArrayRef<const gmx::RVec> shift_vec);

    //! Returns the pair list parameters
    const PairlistParams& params() const { return params_; }

    //! Returns the number of steps performed with the current pair list
    int numStepsWithPairlist(int64_t step) const
    {
        return static_cast<int>(step - outerListCreationStep_);
    }

    //! Returns whether step is a dynamic list pruning step, for CPU lists
    bool isDynamicPruningStepCpu(int64_t step) const
    {
        return (params_.useDynamicPruning && numStepsWithPairlist(step) % params_.nstlistPrune == 0);
    }

    //! Returns whether step is a dynamic list pruning step, for GPU lists
    bool isDynamicPruningStepGpu(int64_t step) const
    {
        const int age = numStepsWithPairlist(step);

        return (params_.useDynamicPruning && age > 0 && age < params_.lifetime
                && step % params_.mtsFactor == 0
                && (params_.haveMultipleDomains_ || age % (2 * params_.mtsFactor) == 0));
    }

    //! Changes the pair-list outer and inner radius
    void changePairlistRadii(real rlistOuter, real rlistInner)
    {
        params_.rlistOuter = rlistOuter;
        params_.rlistInner = rlistInner;
    }

    //! Returns the pair-list set for the given locality
    const PairlistSet& pairlistSet(gmx::InteractionLocality iLocality) const
    {
        if (iLocality == gmx::InteractionLocality::Local)
        {
            return *localSet_;
        }
        else
        {
            GMX_ASSERT(nonlocalSet_, "Need a non-local set when requesting access");
            return *nonlocalSet_;
        }
    }

private:
    //! Returns the pair-list set for the given locality
    PairlistSet& pairlistSet(gmx::InteractionLocality iLocality)
    {
        if (iLocality == gmx::InteractionLocality::Local)
        {
            return *localSet_;
        }
        else
        {
            GMX_ASSERT(nonlocalSet_, "Need a non-local set when requesting access");
            return *nonlocalSet_;
        }
    }

    //! Parameters for the search and list pruning setup
    PairlistParams params_;
    //! Pair list balancing parameter for use with GPU
    int minimumIlistCountForGpuBalancing_;
    //! Local pairlist set
    std::unique_ptr<PairlistSet> localSet_;
    //! Non-local pairlist set
    std::unique_ptr<PairlistSet> nonlocalSet_;
    //! MD step at with the outer lists in pairlistSets_ were created
    int64_t outerListCreationStep_;
};

#endif
