/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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

/*! \file
 * \brief
 * Declares gmx::IOptionValueStore.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_IVALUESTORE_H
#define GMX_OPTIONS_IVALUESTORE_H

namespace gmx
{

template<typename T>
class ArrayRef;

/*! \internal
 * \brief
 * Represents the final storage location of option values.
 *
 * \todo
 * Try to make this more like a write-only interface, getting rid of the need
 * to access the stored values through this interface.  That would simplify
 * things.
 *
 * \ingroup module_options
 */
template<typename T>
class IOptionValueStore
{
public:
    virtual ~IOptionValueStore() {}

    //! Returns the number of values stored so far.
    virtual int valueCount() = 0;
    //! Returns a reference to the actual values.
    virtual ArrayRef<T> values() = 0;
    //! Removes all stored values.
    virtual void clear() = 0;
    //! Reserves memory for additional `count` entries.
    virtual void reserve(size_t count) = 0;
    //! Appends a value to the store.
    virtual void append(const T& value) = 0;
};

} // namespace gmx

#endif
