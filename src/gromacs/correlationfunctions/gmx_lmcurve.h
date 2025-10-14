/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * Declares a driver routine for lmfit.
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_correlationfunctions
 */
#ifndef GMX_CORRELATION_FUNCTIONS_GMX_LMCURVE_H
#define GMX_CORRELATION_FUNCTIONS_GMX_LMCURVE_H
#include "gromacs/correlationfunctions/expfit.h"
/*! \brief function type for passing to fitting routine */
typedef double (*t_lmcurve)(double x, const double* a);
/*! \brief lmfit_exp supports fitting of different functions
 *
 * This routine calls the Levenberg-Marquardt non-linear fitting
 * routine for fitting a data set with errors to a target function.
 * Fitting routines included in gromacs in src/external/lmfit.
 */
bool lmfit_exp(int          nfit,
               const double x[],
               const double y[],
               const double dy[],
               double       parm[],
               bool         bVerbose,
               int          eFitFn,
               int          nfix);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern t_lmcurve lmcurves[effnNR + 1];

#endif
