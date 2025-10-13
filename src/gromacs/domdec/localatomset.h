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

/*! \file
 * \libinternal \brief
 * Declares gmx::LocalAtomSet.
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \inlibraryapi
 * \ingroup module_domdec
 */
#ifndef GMX_DOMDEC_LOCALATOMSET_H
#define GMX_DOMDEC_LOCALATOMSET_H

#include <cstddef>

namespace gmx
{
template<typename>
class ArrayRef;
namespace internal
{
class LocalAtomSetData;
} // namespace internal
/*! \libinternal \brief
 * A local atom set collects local, global and collective indices of
 * the home atoms on a rank. The indices of the home atoms are automatically
 * updated during domain decomposition, thus gmx::LocalAtomSet::localIndex
 * enables iteration over local atoms properties like coordinates or forces.
 * TODO: add a LocalAtomSet iterator.
 *
 * To generate a LocalAtomSet call gmx::LocalAtomSetManger::add and keep the
 * handle to the LocalAtomSet returned from this call.
 *
 * \inlibraryapi
 * \ingroup module_domdec
 */
class LocalAtomSet
{
public:
    friend class LocalAtomSetManager;
    /*! \brief Maps indices on rank [0..numAtomsLocal_) to global atom indicices.
     *
     * \returns the collective index.
     */
    ArrayRef<const int> collectiveIndex() const;
    /*! \brief Global indices of the atoms in this set.
     *
     * \note For best performance, store and use a local copy of the arrayref.
     *
     * \returns the global index.
     */
    ArrayRef<const int> globalIndex() const;
    /*! \brief Local indices of the atoms.
     *
     * For example, the i-th local atom coordinate of this set is
     * x[atomSet.localIndex()[i]].
     *
     * When using in a loop other than a range-based for loop,
     * performance may improve if the ArrayRef is stored in
     * a local variable before the loop is entered.
     * Updated within domain-decomposition.
     *
     * \note For best performance, store and use a local copy of the ArrayRef.
     *
     * \returns the local index.
     */
    ArrayRef<const int> localIndex() const;
    /*! \brief The number of atoms from this group index on this rank.
     *
     * \note For best performance, store and use a local copy of the ArrayRef.
     */
    std::size_t numAtomsLocal() const;
    /*! \brief The number of all atoms from this group index on all ranks together.
     *
     * \note For best performance, store and use a local copy.
     */
    std::size_t numAtomsGlobal() const;

private:
    /*! \brief Constructs a new atom set by setting a reference to its
     * internal data.
     * \param[in] data The data for the atom set is stored
     * in LocalAtomSetData, which is manged by \ref gmx::LocalAtomSetManager.
     */
    explicit LocalAtomSet(const internal::LocalAtomSetData& data);

    const internal::LocalAtomSetData* data_;
};

} // namespace gmx

#endif
