/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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

/*! \libinternal \file
 * \brief Contains functions relevant to simulation setup in MD drivers
 *
 * \inlibraryapi
 * \ingroup module_domdec
 */

#ifndef GMX_DOMDEC_MDSETUP_H
#define GMX_DOMDEC_MDSETUP_H

#include "gromacs/math/vectypes.h"

struct bonded_threading_t;
struct gmx_localtop_t;
struct gmx_mtop_t;
struct gmx_shellfc_t;
struct t_commrec;
struct t_forcerec;
struct t_inputrec;
struct t_mdatoms;

namespace gmx
{
class Constraints;
class ForceBuffers;
class MDAtoms;
class VirtualSitesHandler;

/*! \brief Gets the local shell with domain decomposition
 *
 * \param[in]     cr        Communication record
 * \param[in]     md        The MD atom data
 * \param[in,out] shfc      The shell/flexible-constraint data
 */
void make_local_shells(const t_commrec* cr, const t_mdatoms& md, gmx_shellfc_t* shfc);

/*! \brief Sets atom data for several MD algorithms
 *
 * Most MD algorithms require two different setup calls:
 * one for initialization and parameter setting and one for atom data setup.
 * This routine sets the atom data for the (locally available) atoms.
 * This is called at the start of serial runs and during domain decomposition.
 *
 * \param[in]     cr         Communication record
 * \param[in]     inputrec   Input parameter record
 * \param[in]     top_global The global topology
 * \param[in,out] top        The local topology
 * \param[in,out] fr         The force calculation parameter/data record
 * \param[out]    force      The force buffer
 * \param[out]    mdAtoms    The MD atom data
 * \param[in,out] constr     The constraints handler, can be NULL
 * \param[in,out] vsite      The virtual site data, can be NULL
 * \param[in,out] shellfc    The shell/flexible-constraint data, can be NULL
 */
void mdAlgorithmsSetupAtomData(const t_commrec*     cr,
                               const t_inputrec&    inputrec,
                               const gmx_mtop_t&    top_global,
                               gmx_localtop_t*      top,
                               t_forcerec*          fr,
                               ForceBuffers*        force,
                               MDAtoms*             mdAtoms,
                               Constraints*         constr,
                               VirtualSitesHandler* vsite,
                               gmx_shellfc_t*       shellfc);

} // namespace gmx

#endif
