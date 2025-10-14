/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#ifndef GMX_MDLIB_ENERDATA_UTILS_H
#define GMX_MDLIB_ENERDATA_UTILS_H

#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"

struct gmx_enerdata_t;
struct gmx_grppairener_t;
struct t_lambda;

void reset_dvdl_enerdata(gmx_enerdata_t* enerd);
/* Resets only the dvdl energy data */

void reset_enerdata(gmx_enerdata_t* enerd);
/* Resets the energy data */

/*! \brief Sums energy group pair contributions into epot */
void sum_epot(const gmx_grppairener_t& grpp, real* epot);

/*! \brief Accumulates potential energy contributions to obtain final potential energies
 *
 * Accumulates energy group pair contributions into the output energy components
 * and sums all potential energies into the total potential energy term.
 * With free-energy also computes the foreign lambda potential energy differences.
 *
 * \param[in,out] enerd    Energy data with components to sum and to accumulate into
 * \param[in]     lambda   Vector of free-energy lambdas
 * \param[in]     fepvals  Foreign lambda energy differences, only summed with !=nullptr
 */
void accumulatePotentialEnergies(gmx_enerdata_t*           enerd,
                                 gmx::ArrayRef<const real> lambda,
                                 const t_lambda*           fepvals);

/*! \brief Accumulates kinetic and constraint contributions to dH/dlambda and foreign energies */
void accumulateKineticLambdaComponents(gmx_enerdata_t*           enerd,
                                       gmx::ArrayRef<const real> lambda,
                                       const t_lambda&           fepvals);

#endif
