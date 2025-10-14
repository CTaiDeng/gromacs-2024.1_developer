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
 * \brief This file contains function declarations necessary for
 * mananging the PP side of PME-only ranks.
 *
 * \author Berk Hess <hess@kth.se>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_ewald
 */

#ifndef GMX_EWALD_PME_PP_H
#define GMX_EWALD_PME_PP_H

#include "gromacs/math/vectypes.h"

struct gmx_wallcycle;
struct interaction_const_t;
struct t_commrec;
struct t_forcerec;

class GpuEventSynchronizer;

namespace gmx
{
class ForceWithVirial;
class PmePpCommGpu;
template<typename>
class ArrayRef;
} // namespace gmx

/*! \brief Send the charges and maxshift to out PME-only node. */
void gmx_pme_send_parameters(const t_commrec*           cr,
                             const interaction_const_t& interactionConst,
                             bool                       bFreeEnergy_q,
                             bool                       bFreeEnergy_lj,
                             gmx::ArrayRef<const real>  chargeA,
                             gmx::ArrayRef<const real>  chargeB,
                             gmx::ArrayRef<const real>  sqrt_c6A,
                             gmx::ArrayRef<const real>  sqrt_c6B,
                             gmx::ArrayRef<const real>  sigmaA,
                             gmx::ArrayRef<const real>  sigmaB,
                             int                        maxshift_x,
                             int                        maxshift_y);

/*! \brief Send the coordinates to our PME-only node and request a PME calculation */
void gmx_pme_send_coordinates(t_forcerec*                    fr,
                              const t_commrec*               cr,
                              const matrix                   box,
                              gmx::ArrayRef<const gmx::RVec> x,
                              real                           lambda_q,
                              real                           lambda_lj,
                              bool                           computeEnergyAndVirial,
                              int64_t                        step,
                              bool                           useGpuPmePpComms,
                              bool                           reinitGpuPmePpComms,
                              bool                           sendCoordinatesFromGpu,
                              bool                           receiveForcesToGpu,
                              GpuEventSynchronizer*          coordinatesReadyOnDeviceEvent,
                              bool                           useMdGpuGraph,
                              gmx_wallcycle*                 wcycle);

/*! \brief Tell our PME-only node to finish */
void gmx_pme_send_finish(const t_commrec* cr);

/*! \brief Tell our PME-only node to reset all cycle and flop counters */
void gmx_pme_send_resetcounters(const t_commrec* cr, int64_t step);

/*! \brief PP nodes receive the long range forces from the PME nodes */
void gmx_pme_receive_f(gmx::PmePpCommGpu*    pmePpCommGpu,
                       const t_commrec*      cr,
                       gmx::ForceWithVirial* forceWithVirial,
                       real*                 energy_q,
                       real*                 energy_lj,
                       real*                 dvdlambda_q,
                       real*                 dvdlambda_lj,
                       bool                  useGpuPmePpComms,
                       bool                  receivePmeForceToGpu,
                       float*                pme_cycles);

/*! \brief Tell our PME-only node to switch to a new grid size */
void gmx_pme_send_switchgrid(const t_commrec* cr, ivec grid_size, real ewaldcoeff_q, real ewaldcoeff_lj);

#endif
