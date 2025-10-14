/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2009- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Declares gmx::SelectionCompiler.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_selection
 */
#ifndef GMX_SELECTION_COMPILER_H
#define GMX_SELECTION_COMPILER_H

namespace gmx
{

class SelectionCollection;

/*! \internal
 * \brief
 * Implements selection compilation.
 *
 * This function is used to implement SelectionCollection::compile().
 * It prepares the selections in a selection collection for evaluation and
 * performs some optimizations.
 *
 * Before compilation, the selection collection should have been initialized
 * with gmx_ana_selcollection_parse_*().
 * The compiled selection collection can be passed to
 * gmx_ana_selcollection_evaluate() to evaluate the selection for a frame.
 * If an error occurs, \p coll is cleared.
 *
 * The covered fraction information in \p coll is initialized to
 * \ref CFRAC_NONE.
 *
 * See \ref page_module_selection_compiler.
 *
 * \param[in, out] coll Selection collection to work on.
 *
 * \ingroup module_selection
 */
void compileSelection(SelectionCollection* coll);

} // namespace gmx

#endif
