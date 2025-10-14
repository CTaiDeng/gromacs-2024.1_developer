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

/*! \libinternal \file
 * \brief
 * Declares an error handling interface for key-value tree operations.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_KEYVALUETREEERROR_H
#define GMX_UTILITY_KEYVALUETREEERROR_H

namespace gmx
{

class KeyValueTreePath;
class UserInputError;

class IKeyValueTreeErrorHandler
{
public:
    virtual bool onError(UserInputError* ex, const KeyValueTreePath& context) = 0;

protected:
    virtual ~IKeyValueTreeErrorHandler();
};

//! \cond libapi
/*! \brief
 * Returns a default IKeyValueTreeErrorHandler that throws on first exception.
 *
 * \ingroup module_utility
 */
IKeyValueTreeErrorHandler* defaultKeyValueTreeErrorHandler();
//! \endcond

} // namespace gmx

#endif
