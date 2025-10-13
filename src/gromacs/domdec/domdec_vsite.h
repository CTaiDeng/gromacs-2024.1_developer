/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2005- The GROMACS Authors
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

/*! \internal \file
 *
 * \brief This file declares functions for domdec to use
 * while managing virtual sites.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_DOMDEC_VSITE_H
#define GMX_DOMDEC_DOMDEC_VSITE_H

#include "gromacs/utility/arrayref.h"

struct gmx_domdec_t;
struct InteractionList;

/*! \brief Clears the local indices for the virtual site communication setup */
void dd_clear_local_vsite_indices(struct gmx_domdec_t* dd);

/*! \brief Sets up communication and atom indices for all local vsites */
int dd_make_local_vsites(struct gmx_domdec_t* dd, int at_start, gmx::ArrayRef<InteractionList> lil);

/*! \brief Initializes the data structures for virtual site communication */
void init_domdec_vsites(struct gmx_domdec_t* dd, int n_intercg_vsite);

#endif
