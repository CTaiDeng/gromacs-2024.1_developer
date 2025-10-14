/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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

/*! \libinternal
 * \file
 * \brief
 * Declares routine for computing a cross correlation between two data sets
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \inlibraryapi
 * \ingroup module_correlationfunctions
 */
#ifndef GMX_CROSSCORR_H
#define GMX_CROSSCORR_H

#include "gromacs/utility/real.h"

/*! \brief
 * fft cross correlation algorithm.
 * Computes corr = f (.) g
 *
 * \param[in] n number of data point
 * \param[in] f First function
 * \param[in] g Second function
 * \param[out] corr The cross correlation
 */
void cross_corr(int n, real f[], real g[], real corr[]);

/*! \brief
 * fft cross correlation algorithm.
 *
 * Computes corr[n] = f[n][i] (.) g[n][i], that is for nFunc
 * pairs of arrays n the cross correlation is computed in parallel
 * using OpenMP.
 *
 * \param[in] nFunc nuber of function to crosscorrelate
 * \param[in] nData number of data point in eatch function
 * \param[in] f 2D array of first function to crosscorrelate
 * \param[in] g 2D array of second function to crosscorrelate
 * \param[out] corr 2D array of the cross correlations
 */
void many_cross_corr(int nFunc, int* nData, real** f, real** g, real** corr);

#endif
