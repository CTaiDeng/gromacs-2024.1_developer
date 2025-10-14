/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2023- The GROMACS Authors
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

#ifndef GMX_GMXPREPROCESS_MASSREPARTITIONING_H
#define GMX_GMXPREPROCESS_MASSREPARTITIONING_H

#include "gromacs/utility/real.h"

struct gmx_mtop_t;
class WarningHandler;

namespace gmx
{

/*! \brief Scales the smallest masses in the system by up to \p massFactor
 *
 * First finds the smallest atom mass. Then sets all masses that are smaller
 * than the smallest mass time massFactor to the smallest mass time massFactor.
 * The additional mass is taken away from the atom bound to the light atom.
 * A warning is generated when light atoms are present that are unbound.
 * An error is generated when perturbed masses are affected or when a light
 * atom is bound to multiple other atoms or when a bound atom does becomes
 * lighter than the smallest mass times massFactor.
 *
 * \param[in,out] mtop        The topology to modify
 * \param[in]     useFep      Whether free-energy perturbation is active
 * \param[in]     massFactor  The factor to scale the smallest mass by
 * \param[in,out] wi          Warning handler
 */
void repartitionAtomMasses(gmx_mtop_t* mtop, bool useFep, real massFactor, WarningHandler* wi);

} // namespace gmx

#endif
