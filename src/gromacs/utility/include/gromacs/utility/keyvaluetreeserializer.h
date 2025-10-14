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
 * Declares serialization routines for KeyValueTree objects.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_KEYVALUETREESERIALIZER_H
#define GMX_UTILITY_KEYVALUETREESERIALIZER_H

namespace gmx
{

class KeyValueTreeObject;
class ISerializer;

//! \cond libapi
/*! \brief
 * Serializes a KeyValueTreeObject with given serializer.
 *
 * \ingroup module_utility
 */
void serializeKeyValueTree(const KeyValueTreeObject& root, ISerializer* serializer);
/*! \brief
 * Deserializes a KeyValueTreeObject from a given serializer.
 *
 * \ingroup module_utility
 */
KeyValueTreeObject deserializeKeyValueTree(ISerializer* serializer);
//! \endcond

} // namespace gmx

#endif
