/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 * \brief Declares helper functions for mdrun pertaining to free energy calculations.
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrunutility
 * \inlibraryapi
 */

#ifndef GMX_MDRUNUTILITY_FREEENERGY_H
#define GMX_MDRUNUTILITY_FREEENERGY_H

struct ReplicaExchangeParameters;
struct t_inputrec;

namespace gmx
{

/*! \brief Compute the period at which FEP calculation is performed
 *
 * This harmonizes the free energy calculation period specified by
 * `nstdhdl` with the periods specified by expanded ensemble,
 * replica exchange, and AWH.
 *
 * \param inputrec      The input record
 * \param replExParams  The replica exchange parameters
 * \return              The period required by the involved algorithms
 */
int computeFepPeriod(const t_inputrec& inputrec, const ReplicaExchangeParameters& replExParams);

} // namespace gmx

#endif // GMX_MDRUNUTILITY_FREEENERGY_H
