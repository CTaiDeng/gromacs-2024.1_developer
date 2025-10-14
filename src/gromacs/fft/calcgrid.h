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

#ifndef GMX_FFT_CALCGRID_H
#define GMX_FFT_CALCGRID_H

#include <cstdio>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"

real calcFftGrid(FILE* fp, const matrix box, real gridSpacing, int minGridPointsPerDim, int* nx, int* ny, int* nz);
/* Sets the number of grid points for each zero n* to the first reasonable
 * number which gives a spacing equal to or smaller than gridSpacing
 * and is >= minGridPointsPerDim.
 * Returns the maximum grid spacing.
 */

#endif
