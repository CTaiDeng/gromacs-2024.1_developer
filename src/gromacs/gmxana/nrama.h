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

#ifndef GMX_GMXANA_NRAMA_H
#define GMX_GMXANA_NRAMA_H

#include "gromacs/fileio/trxio.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/real.h"

struct gmx_output_env_t;

typedef struct
{
    gmx_bool bShow;
    char*    label;
    int      iphi, ipsi; /* point in the dih array of xr... */
} t_phipsi;

typedef struct
{
    int  ai[4];
    int  mult;
    real phi0;
    real ang;
} t_dih;

typedef struct
{
    int               ndih;
    t_dih*            dih;
    int               npp;
    t_phipsi*         pp;
    t_trxstatus*      traj;
    int               natoms;
    int               amin, amax;
    real              t;
    rvec*             x;
    matrix            box;
    t_idef*           idef;
    PbcType           pbcType;
    gmx_output_env_t* oenv;
} t_xrama;

t_topology* init_rama(gmx_output_env_t* oenv, const char* infile, const char* topfile, t_xrama* xr, int mult);

gmx_bool new_data(t_xrama* xr);

#endif /* GMX_GMXANA_NRAMA_H */
