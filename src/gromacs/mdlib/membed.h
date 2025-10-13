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

#ifndef GMX_MDLIB_MEMBED_H
#define GMX_MDLIB_MEMBED_H

#include <cstdio>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"

struct gmx_membed_t;
struct gmx_mtop_t;
struct t_commrec;
struct t_filenm;
struct t_inputrec;
class t_state;

/* initialisation of membed code */
gmx_membed_t* init_membed(FILE*          fplog,
                          int            nfile,
                          const t_filenm fnm[],
                          gmx_mtop_t*    mtop,
                          t_inputrec*    inputrec,
                          t_state*       state,
                          t_commrec*     cr,
                          real*          cpt);

/* rescaling the coordinates voor de membed code */
void rescale_membed(int step_rel, gmx_membed_t* membed, rvec* x);

void free_membed(gmx_membed_t* membed);

#endif
