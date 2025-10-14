/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2009- The GROMACS Authors
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

/*! \internal \file
 * \brief API for calculation of centers of mass/geometry.
 *
 * This header defines a few functions that can be used to calculate
 * centers of mass/geometry for a group of atoms.
 * These routines can be used independently of the other parts of the
 * library, but they are also used internally by the selection engine.
 * In most cases, it should not be necessary to call these functions
 * directly.
 * Instead, one should write an analysis tool such that it gets all
 * positions through selections.
 *
 * The functions in the header can be divided into a few groups based on the
 * parameters they take. The simplest group of functions calculates the center
 * of a single group of atoms:
 *  - gmx_calc_cog(): Calculates the center of geometry (COG) of a given
 *    group of atoms.
 *  - gmx_calc_com(): Calculates the center of mass (COM) of a given group
 *    of atoms.
 *  - gmx_calc_comg(): Calculates either the COM or COG, based on a
 *    boolean flag.
 *
 * A second set of routines is provided for calculating the centers for groups
 * that wrap over periodic boundaries (gmx_calc_cog_pbc(), gmx_calc_com_pbc(),
 * gmx_calc_comg_pbc()). These functions are slower, because they need to
 * adjust the center iteratively.
 *
 * It is also possible to calculate centers for several groups of atoms in
 * one call. The functions gmx_calc_cog_block(), gmx_calc_com_block() and
 * gmx_calc_comg_block() take an index group and a partitioning of that index
 * group (as a \c t_block structure), and calculate the centers for
 * each group defined by the \c t_block structure separately.
 *
 * Finally, there is a function gmx_calc_comg_blocka() that takes both the
 * index group and the partitioning as a single \c t_blocka structure.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_selection
 */
#ifndef GMX_SELECTION_CENTEROFMASS_H
#define GMX_SELECTION_CENTEROFMASS_H

#include "gromacs/math/vectypes.h"

struct gmx_mtop_t;
struct t_block;
struct t_blocka;
struct t_pbc;

/*! \brief
 * Calculate a single center of geometry.
 *
 * \param[in]  top    Topology structure (unused, can be NULL).
 * \param[in]  x      Position vectors of all atoms.
 * \param[in]  nrefat Number of atoms in the index.
 * \param[in]  index  Indices of atoms.
 * \param[out] xout   COG position for the indexed atoms.
 */
void gmx_calc_cog(const gmx_mtop_t* top, rvec x[], int nrefat, const int index[], rvec xout);
/** Calculate a single center of mass. */
void gmx_calc_com(const gmx_mtop_t* top, rvec x[], int nrefat, const int index[], rvec xout);
/** Calculate force on a single center of geometry. */
void gmx_calc_cog_f(const gmx_mtop_t* top, rvec f[], int nrefat, const int index[], rvec fout);
/*! \brief
 * Calculate force on a single center of mass.
 *
 * \param[in]  top    Topology structure (unused, can be NULL).
 * \param[in]  f      Forces on all atoms.
 * \param[in]  nrefat Number of atoms in the index.
 * \param[in]  index  Indices of atoms.
 * \param[out] fout   Force on the COM position for the indexed atoms.
 */
void gmx_calc_com_f(const gmx_mtop_t* top, rvec f[], int nrefat, const int index[], rvec fout);
/** Calculate a single center of mass/geometry. */
void gmx_calc_comg(const gmx_mtop_t* top, rvec x[], int nrefat, const int index[], bool bMass, rvec xout);
/** Calculate force on a single center of mass/geometry. */
void gmx_calc_comg_f(const gmx_mtop_t* top, rvec f[], int nrefat, const int index[], bool bMass, rvec fout);

/** Calculate a single center of geometry iteratively, taking PBC into account. */
void gmx_calc_cog_pbc(const gmx_mtop_t* top, rvec x[], const t_pbc* pbc, int nrefat, const int index[], rvec xout);
/** Calculate a single center of mass iteratively, taking PBC into account. */
void gmx_calc_com_pbc(const gmx_mtop_t* top, rvec x[], const t_pbc* pbc, int nrefat, const int index[], rvec xout);
/** Calculate a single center of mass/geometry iteratively with PBC. */
void gmx_calc_comg_pbc(const gmx_mtop_t* top,
                       rvec              x[],
                       const t_pbc*      pbc,
                       int               nrefat,
                       const int         index[],
                       bool              bMass,
                       rvec              xout);

/*! \brief
 * Calculate centers of geometry for a blocked index.
 *
 * \param[in]  top   Topology structure (unused, can be NULL).
 * \param[in]  x     Position vectors of all atoms.
 * \param[in]  block t_block structure that divides \p index into blocks.
 * \param[in]  index Indices of atoms.
 * \param[out] xout  \p block->nr COG positions.
 */
void gmx_calc_cog_block(const gmx_mtop_t* top, rvec x[], const t_block* block, const int index[], rvec xout[]);
/** Calculate centers of mass for a blocked index. */
void gmx_calc_com_block(const gmx_mtop_t* top, rvec x[], const t_block* block, const int index[], rvec xout[]);
/** Calculate forces on centers of geometry for a blocked index. */
void gmx_calc_cog_f_block(const gmx_mtop_t* top, rvec f[], const t_block* block, const int index[], rvec fout[]);
/*! \brief
 * Calculate forces on centers of mass for a blocked index.
 *
 * \param[in]  top   Topology structure (unused, can be NULL).
 * \param[in]  f     Forces on all atoms.
 * \param[in]  block t_block structure that divides \p index into blocks.
 * \param[in]  index Indices of atoms.
 * \param[out] fout  \p block->nr Forces on COM positions.
 */
void gmx_calc_com_f_block(const gmx_mtop_t* top, rvec f[], const t_block* block, const int index[], rvec fout[]);
/** Calculate centers of mass/geometry for a blocked index. */
void gmx_calc_comg_block(const gmx_mtop_t* top,
                         rvec              x[],
                         const t_block*    block,
                         const int         index[],
                         bool              bMass,
                         rvec              xout[]);
/** Calculate forces on centers of mass/geometry for a blocked index. */
void gmx_calc_comg_f_block(const gmx_mtop_t* top,
                           rvec              f[],
                           const t_block*    block,
                           const int         index[],
                           bool              bMass,
                           rvec              fout[]);
/** Calculate centers of mass/geometry for a set of blocks; */
void gmx_calc_comg_blocka(const gmx_mtop_t* top, rvec x[], const t_blocka* block, bool bMass, rvec xout[]);
/** Calculate forces on centers of mass/geometry for a set of blocks; */
void gmx_calc_comg_f_blocka(const gmx_mtop_t* top, rvec x[], const t_blocka* block, bool bMass, rvec xout[]);

#endif
