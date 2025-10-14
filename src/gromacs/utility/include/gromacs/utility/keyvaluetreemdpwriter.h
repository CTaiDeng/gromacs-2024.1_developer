/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * Declares a function to write a flat key-value tree to look like
 * old-style mdp output.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_KEYVALUETREEMDPWRITER_H
#define GMX_UTILITY_KEYVALUETREEMDPWRITER_H

namespace gmx
{

class KeyValueTreeObject;
class TextWriter;

/*! \brief Write a flat key-value \c tree to \c writer in mdp style.
 *
 * Sub-objects will output nothing, so they can be used to
 * contain a special key-value pair to create a comment, as
 * well as the normal key and value. The comment pair will
 * have a key of "comment", and the value will be used as a
 * comment (if non-empty). */
void writeKeyValueTreeAsMdp(TextWriter* writer, const KeyValueTreeObject& tree);

} // namespace gmx

#endif
