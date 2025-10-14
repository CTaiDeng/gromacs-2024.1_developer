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
 *
 * \brief Handling of free energy parameters
 *
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_mdlib
 */

#ifndef GMX_MDLIB_FREEENERGYPARAMETERS_H
#define GMX_MDLIB_FREEENERGYPARAMETERS_H

#include <cstdint>

#include <array>

#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/real.h"

struct t_lambda;

namespace gmx
{

/*! \brief Evaluate the current lambdas
 *
 * \param[in] step the current simulation step
 * \param[in] fepvals describing the lambda setup
 * \param[in] currentLambdaState the lambda state to use to set the lambdas, -1 if not set
 * \returns the current lambda-value array
 */
gmx::EnumerationArray<FreeEnergyPerturbationCouplingType, real> currentLambdas(int64_t         step,
                                                                               const t_lambda& fepvals,
                                                                               int currentLambdaState);

} // namespace gmx

#endif
