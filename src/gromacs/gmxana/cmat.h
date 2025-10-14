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

#ifndef GMX_GMXANA_CMAT_H
#define GMX_GMXANA_CMAT_H

#include "gromacs/utility/real.h"

struct gmx_output_env_t;

typedef struct
{
    int  i, j;
    real dist;
} t_dist;

typedef struct
{
    int conf, clust;
} t_clustid;

struct t_mat
{
    int    n1, nn;
    int*   m_ind;
    bool   b1D;
    real   minrms, maxrms, sumrms;
    real*  erow;
    real** mat;
};

extern t_mat* init_mat(int n1, bool b1D);

extern void copy_t_mat(t_mat* dst, t_mat* src);

extern void reset_index(t_mat* m);

extern void swap_rows(t_mat* m, int iswap, int jswap);

extern void set_mat_entry(t_mat* m, int i, int j, real val);

extern void done_mat(t_mat** m);

extern real mat_energy(t_mat* mat);

extern void low_rmsd_dist(const char* fn, real maxrms, int nn, real** mat, const gmx_output_env_t* oenv);

extern void rmsd_distribution(const char* fn, t_mat* m, const gmx_output_env_t* oenv);

extern t_clustid* new_clustid(int n1);

#endif
