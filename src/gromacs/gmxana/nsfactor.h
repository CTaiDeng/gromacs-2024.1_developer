/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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

#ifndef GMX_GMXANA_NSFACTOR_H
#define GMX_GMXANA_NSFACTOR_H

#include "gromacs/math/vectypes.h"

struct t_topology;

typedef struct gmx_neutron_atomic_structurefactors_t
{
    int     nratoms;
    int*    p;       /* proton number */
    int*    n;       /* neuton number */
    double* slength; /* scattering length in fm */
    char**  atomnm;  /* atom symbol */
} gmx_neutron_atomic_structurefactors_t;

typedef struct gmx_sans_t
{
    const t_topology* top;     /* topology */
    double*           slength; /* scattering length for this topology */
} gmx_sans_t;

typedef struct gmx_radial_distribution_histogram_t
{
    int     grn;      /* number of bins */
    double  binwidth; /* bin size */
    double* r;        /* Distances */
    double* gr;       /* Probability */
} gmx_radial_distribution_histogram_t;

typedef struct gmx_static_structurefactor_t
{
    int     qn;    /* number of items */
    double* s;     /* scattering */
    double* q;     /* q vectors */
    double  qstep; /* q increment */
} gmx_static_structurefactor_t;

void check_binwidth(real binwidth);

void check_mcover(real mcover);

void normalize_probability(int n, double* a);

gmx_neutron_atomic_structurefactors_t* gmx_neutronstructurefactors_init(const char* datfn);

gmx_sans_t* gmx_sans_init(const t_topology* top, gmx_neutron_atomic_structurefactors_t* gnsf);

gmx_radial_distribution_histogram_t* calc_radial_distribution_histogram(gmx_sans_t*  gsans,
                                                                        rvec*        x,
                                                                        matrix       box,
                                                                        const int*   index,
                                                                        int          isize,
                                                                        double       binwidth,
                                                                        bool         bMC,
                                                                        bool         bNORM,
                                                                        real         mcover,
                                                                        unsigned int seed);

gmx_static_structurefactor_t* convert_histogram_to_intensity_curve(gmx_radial_distribution_histogram_t* pr,
                                                                   double start_q,
                                                                   double end_q,
                                                                   double q_step);


#endif
