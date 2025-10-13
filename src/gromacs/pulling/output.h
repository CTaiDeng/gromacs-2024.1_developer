/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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
 *
 * \brief
 * This file declares functions for pull output writing.
 *
 * \author Berk Hess
 *
 * \inlibraryapi
 */

#ifndef GMX_PULLING_OUTPUT_H
#define GMX_PULLING_OUTPUT_H

#include "gromacs/utility/basedefinitions.h"

struct pull_t;
struct gmx_output_env_t;
struct ObservablesHistory;
struct t_filenm;

namespace gmx
{
enum class StartingBehavior;
}

/*! \brief Set up and open the pull output files, when requested.
 *
 * NOTE: This should only be called on the main rank and only when
 *       doing dynamics (e.g. not with energy minimization).
 *
 * \param pull        The pull work data struct
 * \param nfile       Number of files.
 * \param fnm         Standard filename struct.
 * \param oenv        Output options.
 * \param startingBehavior  Describes whether this is a restart appending to output files
 */
void init_pull_output_files(pull_t*                 pull,
                            int                     nfile,
                            const t_filenm          fnm[],
                            const gmx_output_env_t* oenv,
                            gmx::StartingBehavior   startingBehavior);

/*! \brief Print the pull output (x and/or f)
 *
 * \param pull     The pull data structure.
 * \param step     Time step number.
 * \param time     Time.
 */
void pull_print_output(pull_t* pull, int64_t step, double time);

/*! \brief Allocate and initialize pull work history (for average pull output) and set it in a pull work struct
 *
 * \param pull                The pull work struct
 * \param observablesHistory  Container of history data, e.g., pull history.
 */
void initPullHistory(pull_t* pull, ObservablesHistory* observablesHistory);

#endif
