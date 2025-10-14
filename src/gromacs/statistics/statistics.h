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

/*! \libinternal \file
 * \brief
 * Declares simple statistics toolbox
 *
 * \authors David van der Spoel <david.vanderspoel@icm.uu.se>
 * \inlibraryapi
 */
#ifndef GMX_STATISTICS_H
#define GMX_STATISTICS_H

#include <cstdio>

#include <tuple>

#include "gromacs/utility/real.h"

//! Abstract container type
typedef struct gmx_stats* gmx_stats_t;

//! Enum for statistical weights
enum
{
    elsqWEIGHT_NONE,
    elsqWEIGHT_X,
    elsqWEIGHT_Y,
    elsqWEIGHT_XY,
    elsqWEIGHT_NR
};

/*! \brief
 * Initiate a data structure
 * \return the data structure
 */
gmx_stats_t gmx_stats_init();

/*! \brief
 * Destroy a data structure
 * \param stats The data structure
 */
void gmx_stats_free(gmx_stats_t stats);

/*! \brief
 * Add a point to the data set
 * \param[in] stats The data structure
 * \param[in] x   The x value
 * \param[in] y   The y value
 * \param[in] dx  The error in the x value
 * \param[in] dy  The error in the y value
 */
void gmx_stats_add_point(gmx_stats_t stats, double x, double y, double dx, double dy);

/*! \brief
 * Fit the data to y = ax + b, possibly weighted, if uncertainties
 * have been input. da and db may be NULL.
 * \param[in] stats The data structure
 * \param[in] weight type of weighting
 * \param[out] a slope
 * \param[out] b intercept
 * \param[out] da sigma in a
 * \param[out] db sigma in b
 * \param[out] chi2 normalized quality of fit
 * \param[out] Rfit correlation coefficient
 */
void gmx_stats_get_ab(gmx_stats_t stats, int weight, real* a, real* b, real* da, real* db, real* chi2, real* Rfit);

/*! \brief
 * Computes and returns the average value.
 * \param[in]  stats The data structure
 * \return Average value
 * \throws  InconsistentInputError if given no points to average
 */
real gmx_stats_get_average(gmx_stats_t stats);

/*! \brief
 * Pointers may be null, in which case no assignment will be done.
 * \param[in]  stats The data structure
 * \return Tuple of (average value, its standard deviation, its standard error)
 * \throws  InconsistentInputError if given no points to analyze
 */
std::tuple<real, real, real> gmx_stats_get_ase(gmx_stats_t stats);

/****************************************************
 * Some statistics utilities for convenience: useful when a complete data
 * set is available already from another source, e.g. an xvg file.
 ****************************************************/

/*! \brief
 * Fit a straight line y=ax+b thru the n data points x, y.
 * \param[in] n number of points
 * \param[in] x data points x
 * \param[in] y data point y
 * \param[out] a slope
 * \param[out] b intercept
 * \param[out] r correlation coefficient
 * \param[out] chi2 quality of fit
 *
 * \throws  InconsistentInputError if given no points to fit
 */
void lsq_y_ax_b(int n, real x[], real y[], real* a, real* b, real* r, real* chi2);

/*! \copydoc lsq_y_ax_b
 * Suits cases where x is already always computed in double precision
 * even in a mixed-precision build configuration.
 */
void lsq_y_ax_b_xdouble(int n, double x[], real y[], real* a, real* b, real* r, real* chi2);

/*! \brief
 * Fit a straight line y=ax+b thru the n data points x, y.
 * \param[in] n number of points
 * \param[in] x data points x
 * \param[in] y data point y
 * \param[in] dy uncertainty in data point y
 * \param[out] a slope
 * \param[out] b intercept
 * \param[out] da error in slope
 * \param[out] db error in intercept
 * \param[out] r correlation coefficient
 * \param[out] chi2 quality of fit
 *
 * \throws  InconsistentInputError if given no points to fit
 */
void lsq_y_ax_b_error(int n, real x[], real y[], real dy[], real* a, real* b, real* da, real* db, real* r, real* chi2);

#endif
