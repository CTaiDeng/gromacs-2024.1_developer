/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 *
 * \brief
 * Contains datatypes and function declarations needed by AWH to
 * have its force correlation data checkpointed.
 *
 * \author Viveca Lindahl
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_awh
 */

#ifndef GMX_AWH_CORRELATIONHISTORY_H
#define GMX_AWH_CORRELATIONHISTORY_H

struct t_commrec;

namespace gmx
{
class CorrelationGrid;
struct CorrelationGridHistory;

/*! \brief
 * Allocate a correlation grid history with the same structure as the given correlation grid.
 *
 * This function would be called at the start of a new simulation.
 * Note that only sizes and memory are initialized here.
 * History data is set by \ref updateCorrelationGridHistory.
 *
 * \param[in,out] corrGrid      Correlation grid state to initialize with.
 * \returns the correlation grid history struct.
 */
CorrelationGridHistory initCorrelationGridHistoryFromState(const CorrelationGrid& corrGrid);

/*! \brief
 * Restores the correlation grid state from the correlation grid history.
 *
 * \param[in] corrGridHist  Correlation grid history to read.
 * \param[in,out] corrGrid  Correlation grid state to set.
 */
void restoreCorrelationGridStateFromHistory(const CorrelationGridHistory& corrGridHist,
                                            CorrelationGrid*              corrGrid);

/*! \brief
 * Update the correlation grid history for checkpointing.
 *
 * \param[in,out] corrGridHist  Correlation grid history to set.
 * \param[in]     corrGrid      Correlation grid state to read.
 */
void updateCorrelationGridHistory(CorrelationGridHistory* corrGridHist, const CorrelationGrid& corrGrid);

} // namespace gmx

#endif /* GMX_AWH_CORRELATIONHISTORY_H */
