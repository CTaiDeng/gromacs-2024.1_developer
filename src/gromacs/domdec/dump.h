/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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
 *
 * \brief This file declares functions for DD to write PDB files
 * e.g. when reporting problems.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_DUMP_H
#define GMX_DOMDEC_DUMP_H

#include <cstdint>

#include "gromacs/math/vectypes.h"

struct gmx_ddbox_t;
struct gmx_domdec_t;
struct gmx_mtop_t;
struct t_commrec;

//! Write the DD grid to a PDB file
void write_dd_grid_pdb(const char* fn, int64_t step, gmx_domdec_t* dd, matrix box, gmx_ddbox_t* ddbox);

/*! \brief Dump a pdb file with the current DD home + communicated atoms.
 *
 * When natoms=-1, dump all known atoms.
 */
void write_dd_pdb(const char*       fn,
                  int64_t           step,
                  const char*       title,
                  const gmx_mtop_t& mtop,
                  const t_commrec*  cr,
                  int               natoms,
                  const rvec        x[],
                  const matrix      box);

#endif
