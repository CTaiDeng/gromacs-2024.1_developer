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

#ifndef GMX_MATH_VECDUMP_H
#define GMX_MATH_VECDUMP_H

#include <cstdio>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"

void pr_ivec(FILE* fp, int indent, const char* title, const int vec[], int n, gmx_bool bShowNumbers);
void pr_ivec_block(FILE* fp, int indent, const char* title, const int vec[], int n, gmx_bool bShowNumbers);
void pr_ivecs(FILE* fp, int indent, const char* title, const ivec vec[], int n, gmx_bool bShowNumbers);
void pr_rvec(FILE* fp, int indent, const char* title, const real vec[], int n, gmx_bool bShowNumbers);
void pr_fvec(FILE* fp, int indent, const char* title, const float vec[], int n, gmx_bool bShowNumbers);
void pr_dvec(FILE* fp, int indent, const char* title, const double vec[], int n, gmx_bool bShowNumbers);
void pr_rvecs(FILE* fp, int indent, const char* title, const rvec vec[], int n);

#endif
