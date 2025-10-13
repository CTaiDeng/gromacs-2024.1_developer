/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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

#ifndef GMX_GMXANA_CLUSTER_METHODS_H
#define GMX_GMXANA_CLUSTER_METHODS_H

#include <cstdio>

#include "gromacs/utility/real.h"

struct gmx_output_env_t;
struct t_mat;

struct t_clusters
{
    int  ncl;
    int* cl;
};

struct t_nnb
{
    int  nr;
    int* nb;
};


void mc_optimize(FILE*             log,
                 t_mat*            m,
                 real*             time,
                 int               maxiter,
                 int               nrandom,
                 int               seed,
                 real              kT,
                 const char*       conv,
                 gmx_output_env_t* oenv);

void gather(t_mat* m, real cutoff, t_clusters* clust);

void jarvis_patrick(int n1, real** mat, int M, int P, real rmsdcut, t_clusters* clust);

void gromos(int n1, real** mat, real rmsdcut, t_clusters* clust);

#endif
