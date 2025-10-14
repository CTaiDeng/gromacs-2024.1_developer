/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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
 * \brief
 * Declares PreprocessingBondAtomType.
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inlibraryapi
 * \ingroup module_preprocessing
 */
#ifndef GMX_GMXPREPROCESS_GPP_BOND_ATOMTYPE_H
#define GMX_GMXPREPROCESS_GPP_BOND_ATOMTYPE_H

#include <cstdio>

#include <memory>
#include <optional>
#include <string>

/*! \libinternal \brief
 * Storage for all bonded atomtypes during simulation preprocessing.
 */
class PreprocessingBondAtomType
{
public:
    PreprocessingBondAtomType();
    ~PreprocessingBondAtomType();

    //! Get number of defined bond atom types.
    size_t size() const;

    /*! \brief
     * Get name of atom from internal bond atom type number.
     *
     * \param[in] nt Internal number of atom type.
     * \returns The optional type name.
     */
    std::optional<std::string> atomNameFromBondAtomType(int nt) const;

    /*! \brief
     *  Get bond atom type index for atom type name if present in the database, or NOTSET.
     *
     *  \todo The code should be changed to instead use a gmx::compat version
     *  of std::optional to return a handle to the element being searched,
     *  or an empty optional construct if the entry has not been found.
     *
     *  \param[in] str Input string to search type for.
     *  \returns Optional atomtype as integer.
     */
    std::optional<int> bondAtomTypeFromName(const std::string& str) const;

    /*! \brief Add a unique type to the database.
     *
     * \param[in] name Atom name.
     * \returns Index to the type in the database. If the type shares
     *          a name with an existing type, return the index of that type.
     */
    int addBondAtomType(const std::string& name);

    /*! \brief
     * If a value is within the range of the current types or not.
     *
     * \param[in] nt Value to check.
     * \returns True if value is in range.
     */
    bool isSet(int nt) const;

private:
    class Impl;
    //! Pimpl that holds the data.
    std::unique_ptr<Impl> impl_;
};

#endif
