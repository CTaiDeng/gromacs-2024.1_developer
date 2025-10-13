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

#ifndef GMX_MATH_VECCOMPARE_H
#define GMX_MATH_VECCOMPARE_H

#include <cstdio>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

void cmp_rvec(FILE* fp, const char* s, int index, const rvec i1, const rvec i2, real ftol, real abstol);

void cmp_ivec(FILE* fp, const char* s, int index, const ivec i1, const ivec i2);

void cmp_rvecs(FILE* fp, const char* title, int n, const rvec x1[], const rvec x2[], gmx_bool bRMSD, real ftol, real abstol);

#endif
