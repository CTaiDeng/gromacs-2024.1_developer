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

#ifndef GMX_MATH_DO_FIT_H
#define GMX_MATH_DO_FIT_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

real calc_similar_ind(gmx_bool bRho, int nind, const int* index, const real mass[], rvec x[], rvec xp[]);
/* Returns RMSD or Rho (depending on bRho) over all atoms in index */

real rmsdev_ind(int nind, int index[], real mass[], rvec x[], rvec xp[]);
/* Returns the RMS Deviation betweem x and xp over all atoms in index */

real rmsdev(int natoms, real mass[], rvec x[], rvec xp[]);
/* Returns the RMS Deviation betweem x and xp over all atoms */

real rhodev_ind(int nind, int index[], real mass[], rvec x[], rvec xp[]);
/* Returns size-independent Rho similarity parameter over all atoms in index
 * Maiorov & Crippen, PROTEINS 22, 273 (1995).
 */

real rhodev(int natoms, real mass[], rvec x[], rvec xp[]);
/* Returns size-independent Rho similarity parameter over all atoms
 * Maiorov & Crippen, PROTEINS 22, 273 (1995).
 */

void calc_fit_R(int ndim, int natoms, const real* w_rls, const rvec* xp, rvec* x, matrix R);
/* Calculates the rotation matrix R for which
 * sum_i w_rls_i (xp_i - R x_i).(xp_i - R x_i)
 * is minimal. ndim=3 gives full fit, ndim=2 gives xy fit.
 * This matrix is also used do_fit.
 * x_rotated[i] = sum R[i][j]*x[j]
 */

void do_fit_ndim(int ndim, int natoms, real* w_rls, const rvec* xp, rvec* x);
/* Do a least squares fit of x to xp. Atoms which have zero mass
 * (w_rls[i]) are not taken into account in fitting.
 * This makes is possible to fit eg. on Calpha atoms and orient
 * all atoms. The routine only fits the rotational part,
 * therefore both xp and x should be centered round the origin.
 */

void do_fit(int natoms, real* w_rls, const rvec* xp, rvec* x);
/* Calls do_fit with ndim=3, thus fitting in 3D */

void reset_x_ndim(int ndim, int ncm, const int* ind_cm, int nreset, const int* ind_reset, rvec x[], const real mass[]);
/* Put the center of mass of atoms in the origin for dimensions 0 to ndim.
 * The center of mass is computed from the index ind_cm.
 * When ind_cm!=NULL the COM is determined using ind_cm.
 * When ind_cm==NULL the COM is determined for atoms 0 to ncm.
 * When ind_reset!=NULL the coordinates indexed by ind_reset are reset.
 * When ind_reset==NULL the coordinates up to nreset are reset.
 */

void reset_x(int ncm, const int* ind_cm, int nreset, const int* ind_reset, rvec x[], const real mass[]);
/* Calls reset_x with ndim=3, thus resetting all dimesions */

#endif
