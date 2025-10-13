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

#ifndef GMX_GMXLIB_NONBONDED_NONBONDED_H
#define GMX_GMXLIB_NONBONDED_NONBONDED_H

#define GMX_NONBONDED_DO_FORCE (1 << 1)
#define GMX_NONBONDED_DO_SHIFTFORCE (1 << 2)
#define GMX_NONBONDED_DO_FOREIGNLAMBDA (1 << 3)
#define GMX_NONBONDED_DO_POTENTIAL (1 << 4)
#define GMX_NONBONDED_DO_SR (1 << 5)

#endif
