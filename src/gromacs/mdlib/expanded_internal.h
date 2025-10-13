/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

/*! \internal \file
 * \brief Declares internal functionality for expanded ensemble
 *
 * This file is only used by expanded.cpp and tests/expanded.cpp.
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \author Michael Shirts <michael.shirts@colorado.edu>
 * \ingroup module_mdlib
 */
#ifndef GMX_MDLIB_EXPANDEDINTERNAL_H
#define GMX_MDLIB_EXPANDEDINTERNAL_H

#include "gromacs/utility/real.h"

enum class LambdaWeightCalculation : int;

namespace gmx
{
/*! \brief Calculates the acceptance weight for a lambda state transition
 *
 * \param[in] calculationMode  How the lambda weights are calculated
 * \param[in] lambdaEnergyDifference  The difference in energy between the two states
 * \return  The acceptance weight
 */
real calculateAcceptanceWeight(LambdaWeightCalculation calculationMode, real lambdaEnergyDifference);
} // namespace gmx

#endif // GMX_MDLIB_EXPANDEDINTERNAL_H
