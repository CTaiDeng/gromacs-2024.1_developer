/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 * \brief Implements internal functionality for expanded ensemble
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \author Michael Shirts <michael.shirts@colorado.edu>
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "expanded_internal.h"

#include <cmath>

#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/exceptions.h"

namespace gmx
{
real calculateAcceptanceWeight(LambdaWeightCalculation calculationMode, real lambdaEnergyDifference)
{
    if (calculationMode == LambdaWeightCalculation::Barker
        || calculationMode == LambdaWeightCalculation::Minvar)
    {
        /* Barker acceptance rule forumula is used for accumulation of probability for
         * both the Barker variant of the weight accumulation algorithm and the
         * minimum variance variant of the weight accumulation algorithm.
         *
         * Barker acceptance rule for a jump from state i -> j is defined as
         *     exp(-E_i)/exp(-Ei)+exp(-Ej) =   1 / (1 + exp(dE_ij))
         * where dE_ij is the potential energy difference between the two states
         * plus a constant offset that can be removed at the end for numerical stability.
         *     dE_ij = FE_j - FE_i + offset
         * Numerically, this computation can be unstable if dE gets large. (See #3304)
         * To avoid numerical instability, we're calculating it as
         *     1 / (1 + exp(dE_ij))             (if dE < 0)
         *     exp(-dE_ij) / (exp(-dE_ij) + 1)  (if dE > 0)
         */
        if (lambdaEnergyDifference < 0)
        {
            return 1.0 / (1.0 + std::exp(lambdaEnergyDifference));
        }
        else
        {
            return std::exp(-lambdaEnergyDifference) / (1.0 + std::exp(-lambdaEnergyDifference));
        }
    }
    else if (calculationMode == LambdaWeightCalculation::Metropolis)
    {
        /* Metropolis acceptance rule for a jump from state i -> j is defined as
         *     1            (if dE_ij < 0)
         *     exp(-dE_ij)  (if dE_ij >= 0)
         * where dE_ij is the potential energy difference between the two states
         * plus a free energy offset that can be subtracted off later:
         *     dE_ij = FE_j - FE_i + offset
         */
        if (lambdaEnergyDifference < 0)
        {
            return 1.0;
        }
        else
        {
            return std::exp(-lambdaEnergyDifference);
        }
    }

    GMX_THROW(NotImplementedError("Unknown acceptance calculation mode"));
}

} // namespace gmx
