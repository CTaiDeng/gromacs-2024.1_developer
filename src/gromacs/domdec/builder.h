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

/*! \libinternal \file
 *
 * \brief This file declares a builder class for the manager
 * of domain decomposition
 *
 * \author Berk Hess <hess@kth.se>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inlibraryapi
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_BUILDER_H
#define GMX_DOMDEC_BUILDER_H

#include <memory>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"

struct gmx_domdec_t;
struct gmx_mtop_t;
struct gmx_localtop_t;
struct t_commrec;
struct t_inputrec;
class t_state;

namespace gmx
{
class MDLogger;
class LocalAtomSetManager;
class RangePartitioning;
struct DomdecOptions;
struct MdrunOptions;
struct MDModulesNotifiers;
class ObservablesReducerBuilder;

template<typename T>
class ArrayRef;

/*! \libinternal
 * \brief Builds a domain decomposition management object
 *
 * This multi-phase construction needs first a decision about the
 * duty(s) of each rank, and then perhaps to be advised of GPU streams
 * for transfer operations. */
class DomainDecompositionBuilder
{
public:
    //! Constructor
    DomainDecompositionBuilder(const MDLogger&                   mdlog,
                               t_commrec*                        cr,
                               const DomdecOptions&              options,
                               const MdrunOptions&               mdrunOptions,
                               const gmx_mtop_t&                 mtop,
                               const t_inputrec&                 ir,
                               const MDModulesNotifiers&         notifiers,
                               const matrix                      box,
                               ArrayRef<const RangePartitioning> updateGroupingPerMoleculeType,
                               bool                              useUpdateGroups,
                               real                              maxUpdateGroupRadius,
                               ArrayRef<const RVec>              xGlobal,
                               bool                              useGpuForNonbonded,
                               bool                              useGpuForPme,
                               bool                              useGpuForUpdate,
                               bool                              useGpuDirectHalo,
                               bool                              canUseGpuPmeDecomposition);
    //! Destructor
    ~DomainDecompositionBuilder();
    //! Build the resulting DD manager
    std::unique_ptr<gmx_domdec_t> build(LocalAtomSetManager*       atomSets,
                                        const gmx_localtop_t&      localTopology,
                                        const t_state&             localState,
                                        ObservablesReducerBuilder* observablesReducerBuilder);

private:
    class Impl;
    //! Pimpl to hide implementation details
    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
