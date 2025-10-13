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

#ifndef GMX_MDLIB_RBIN_H
#define GMX_MDLIB_RBIN_H

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

struct t_commrec;

typedef struct
{
    int     nreal;
    int     maxreal;
    double* rbuf;
} t_bin;

t_bin* mk_bin();
/* Create a real bin */

void destroy_bin(t_bin* b);
/* Destroy the bin structure */

void reset_bin(t_bin* b);
/* Reset number of entries to zero */

int add_binr(t_bin* b, int nr, const real r[]);
int add_binr(t_bin* b, gmx::ArrayRef<const real> r);
int add_bind(t_bin* b, int nr, const double r[]);
int add_bind(t_bin* b, gmx::ArrayRef<const double> r);
/* Add reals to the bin. Returns index */

void sum_bin(t_bin* b, const t_commrec* cr);
/* Globally sum the reals in the bin */

void extract_binr(t_bin* b, int index, int nr, real r[]);
void extract_binr(t_bin* b, int index, gmx::ArrayRef<real> r);
void extract_bind(t_bin* b, int index, int nr, double r[]);
void extract_bind(t_bin* b, int index, gmx::ArrayRef<double> r);
/* Extract values from the bin, starting from index (see add_bin) */

#endif
