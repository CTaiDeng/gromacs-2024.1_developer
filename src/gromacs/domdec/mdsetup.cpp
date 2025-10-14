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

#include "gmxpre.h"

#include "mdsetup.h"

#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/localtopology.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdlib/wholemoleculetransform.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcebuffers.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

/* TODO: Add a routine that collects the initial setup of the algorithms.
 *
 * The final solution should be an MD algorithm base class with methods
 * for initialization and atom-data setup.
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
                               gmx_shellfc_t*       shellfc)
{
    bool usingDomDec = haveDDAtomOrdering(*cr);

    int numAtomIndex;
    int numHomeAtoms;
    int numTotalAtoms;

    if (usingDomDec)
    {
        numAtomIndex  = dd_natoms_mdatoms(*cr->dd);
        numHomeAtoms  = dd_numHomeAtoms(*cr->dd);
        numTotalAtoms = dd_natoms_mdatoms(*cr->dd);
    }
    else
    {
        numAtomIndex  = -1;
        numHomeAtoms  = top_global.natoms;
        numTotalAtoms = top_global.natoms;
    }

    if (force != nullptr)
    {
        force->resize(numTotalAtoms);
    }

    atoms2md(top_global,
             inputrec,
             numAtomIndex,
             usingDomDec ? cr->dd->globalAtomIndices : std::vector<int>(),
             numHomeAtoms,
             mdAtoms);

    t_mdatoms* mdatoms = mdAtoms->mdatoms();
    if (!usingDomDec)
    {
        gmx_mtop_generate_local_top(top_global, top, inputrec.efep != FreeEnergyPerturbationType::No);
    }

    if (fr->wholeMoleculeTransform && usingDomDec)
    {
        fr->wholeMoleculeTransform->updateAtomOrder(cr->dd->globalAtomIndices, *cr->dd->ga2la);
    }

    if (vsite)
    {
        vsite->setVirtualSites(top->idef.il, mdatoms->nr, mdatoms->homenr, mdatoms->ptype);
    }

    /* Note that with DD only flexible constraints, not shells, are supported
     * and these don't require setup in make_local_shells().
     *
     * TODO: This should only happen in ShellFCElement (it is called directly by the modular
     *       simulator ShellFCElement already, but still used here by legacy simulators)
     */
    if (!usingDomDec && shellfc)
    {
        make_local_shells(cr, *mdatoms, shellfc);
    }

    for (auto& listedForces : fr->listedForces)
    {
        listedForces.setup(top->idef, fr->natoms_force, fr->listedForcesGpu != nullptr);
    }

    if ((usingPme(fr->ic->eeltype) || usingLJPme(fr->ic->vdwtype)) && (cr->duty & DUTY_PME))
    {
        /* This handles the PP+PME rank case where fr->pmedata is valid.
         * For PME-only ranks, gmx_pmeonly() has its own call to gmx_pme_reinit_atoms().
         */
        const int numPmeAtoms = numHomeAtoms - fr->n_tpi;
        gmx_pme_reinit_atoms(fr->pmedata, numPmeAtoms, mdatoms->chargeA, mdatoms->chargeB);
    }

    if (constr)
    {
        constr->setConstraints(top,
                               mdatoms->nr,
                               mdatoms->homenr,
                               mdatoms->massT,
                               mdatoms->invmass,
                               mdatoms->nMassPerturbed != 0,
                               mdatoms->lambda,
                               mdatoms->cFREEZE);
    }
}

} // namespace gmx
