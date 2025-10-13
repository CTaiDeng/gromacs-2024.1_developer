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

/*! \libinternal \file
 * \brief
 * Declares gmx::LocalAtomSetManager
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \inlibraryapi
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_LOCALATOMSETMANAGER_H
#define GMX_DOMDEC_LOCALATOMSETMANAGER_H

#include <memory>

#include "gromacs/utility/basedefinitions.h"

class gmx_ga2la_t;

namespace gmx
{
template<typename>
class ArrayRef;
class LocalAtomSet;

/*! \libinternal \brief
 * Hands out handles to local atom set indices and triggers index recalculation
 * for all sets upon domain decomposition if run in parallel.
 *
 * \inlibraryapi
 * \ingroup module_domdec
 */
class LocalAtomSetManager
{
public:
    LocalAtomSetManager();
    ~LocalAtomSetManager();
#ifndef DOXYGEN
    /*! \brief Add a new atom set to be managed and give back a handle.
     *
     * \todo remove this routine once all indices are represented as
     *       gmx::Index instead of int.
     *
     * \note Not created if the internal int type does match index
     *
     * \tparam T template parameter to use SFINAE for conditional function
     *           activation
     * \tparam U template parameter for conditional function activation
     *
     * \param[in] globalAtomIndex Indices of the atoms to be managed
     * \returns Handle to LocalAtomSet.
     */
    template<typename T = void, typename U = std::enable_if_t<!std::is_same_v<int, Index>, T>>
    LocalAtomSet add(ArrayRef<const int> globalAtomIndex);
#endif
    /*! \brief Add a new atom set to be managed and give back a handle.
     *
     * \param[in] globalAtomIndex Indices of the atoms to be managed
     * \returns Handle to LocalAtomSet.
     */
    LocalAtomSet add(ArrayRef<const Index> globalAtomIndex);

    /*! \brief Recalculate local and collective indices from ga2la.
     * Uses global atom to local atom lookup structure to
     * update atom indices.
     */
    void setIndicesInDomainDecomposition(const gmx_ga2la_t& ga2la);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
