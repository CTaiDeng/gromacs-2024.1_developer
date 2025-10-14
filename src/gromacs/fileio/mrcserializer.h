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
 * \brief
 * Serialization routines for volume data format mrc.
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \ingroup module_fileio
 */

#ifndef GMX_FILEIO_MRCSERIALIZER_H
#define GMX_FILEIO_MRCSERIALIZER_H
namespace gmx
{
class ISerializer;
struct MrcDensityMapHeader;

/*! \brief Serializes an MrcDensityMapHeader from a given serializer.
 * \param[in] serializer the serializer
 * \param[in] mrcHeader file header to be serialized
 */
void serializeMrcDensityMapHeader(ISerializer* serializer, const MrcDensityMapHeader& mrcHeader);
/*! \brief Deserializes an MrcDensityMapHeader from a given serializer.
 * \param[in] serializer the serializer
 * \returns mrc density map header
 */
MrcDensityMapHeader deserializeMrcDensityMapHeader(ISerializer* serializer);
} // namespace gmx
#endif /* end of include guard: GMX_FILEIO_MRCSERIALIZER_H */
