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

#ifndef GMX_GMXANA_GMX_ANA_H
#    define GMX_GMXANA_GMX_ANA_H

int gmx_analyze(int argc, char* argv[]);

int gmx_anaeig(int argc, char* argv[]);

int gmx_awh(int argc, char* argv[]);

int gmx_g_angle(int argc, char* argv[]);

int gmx_bar(int argc, char* argv[]);

int gmx_bundle(int argc, char* argv[]);

int gmx_chi(int argc, char* argv[]);

int gmx_cluster(int argc, char* argv[]);

int gmx_confrms(int argc, char* argv[]);

int gmx_covar(int argc, char* argv[]);

int gmx_current(int argc, char* argv[]);

int gmx_density(int argc, char* argv[]);

int gmx_densmap(int argc, char* argv[]);

int gmx_densorder(int argc, char* argv[]);

int gmx_dielectric(int argc, char* argv[]);

int gmx_dipoles(int argc, char* argv[]);

int gmx_disre(int argc, char* argv[]);

int gmx_dos(int argc, char* argv[]);

int gmx_dyecoupl(int argc, char* argv[]);

int gmx_enemat(int argc, char* argv[]);

int gmx_energy(int argc, char* argv[]);

int gmx_lie(int argc, char* argv[]);

int gmx_filter(int argc, char* argv[]);

int gmx_gyrate(int argc, char* argv[]);

int gmx_h2order(int argc, char* argv[]);

int gmx_hbond(int argc, char* argv[]);

int gmx_helix(int argc, char* argv[]);

int gmx_helixorient(int argc, char* argv[]);

int gmx_hydorder(int argc, char* argv[]);

int gmx_make_edi(int argc, char* argv[]);

int gmx_mindist(int argc, char* argv[]);

int gmx_nmeig(int argc, char* argv[]);

int gmx_nmens(int argc, char* argv[]);

int gmx_nmr(int argc, char* argv[]);

int gmx_nmtraj(int argc, char* argv[]);

int gmx_order(int argc, char* argv[]);

int gmx_polystat(int argc, char* argv[]);

int gmx_potential(int argc, char* argv[]);

int gmx_principal(int argc, char* argv[]);

int gmx_rama(int argc, char* argv[]);

int gmx_rotmat(int argc, char* argv[]);

int gmx_rms(int argc, char* argv[]);

int gmx_rmsdist(int argc, char* argv[]);

int gmx_rmsf(int argc, char* argv[]);

int gmx_rotacf(int argc, char* argv[]);

int gmx_saltbr(int argc, char* argv[]);

int gmx_sham(int argc, char* argv[]);

int gmx_sigeps(int argc, char* argv[]);

int gmx_sorient(int argc, char* argv[]);

int gmx_spol(int argc, char* argv[]);

int gmx_spatial(int argc, char* argv[]);

int gmx_tcaf(int argc, char* argv[]);

int gmx_traj(int argc, char* argv[]);

int gmx_trjorder(int argc, char* argv[]);

int gmx_velacc(int argc, char* argv[]);

int gmx_clustsize(int argc, char* argv[]);

int gmx_mdmat(int argc, char* argv[]);

int gmx_vanhove(int argc, char* argv[]);

int gmx_wham(int argc, char* argv[]);

int gmx_wheel(int argc, char* argv[]);

int gmx_xpm2ps(int argc, char* argv[]);

int gmx_sans(int argc, char* argv[]);

int gmx_saxs(int argc, char* argv[]);

#endif
/* _gmx_ana_h */
