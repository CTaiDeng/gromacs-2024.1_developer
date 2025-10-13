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

#ifndef GMX_MAXWELL_VELOCITIES
#define GMX_MAXWELL_VELOCITIES

#include <cstdio>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"

struct gmx_mtop_t;

namespace gmx
{
class MDLogger;
}

/*! \brief
 * Generate Maxwellian velocities.
 *
 * \param[in] tempi Temperature to generate around
 * \param[in] seed  Random number generator seed. A new one is
 *                  generated if this is -1.
 * \param[in] mtop  Molecular Topology
 * \param[out] v    Velocities
 * \param[in] logger Handle to logging interface.
 */
void maxwell_speed(real tempi, int seed, gmx_mtop_t* mtop, rvec v[], const gmx::MDLogger& logger);

/*! \brief
 * Remove the center of mass motion in a set of coordinates.
 *
 * \param[in]  logger Handle to logging interface.
 * \param[in]  natoms Number of atoms
 * \param[in]  mass   Atomic masses
 * \param[in]  x      Coordinates
 * \param[out] v      Velocities
 */
void stop_cm(const gmx::MDLogger& logger, int natoms, real mass[], rvec x[], rvec v[]);

#endif
