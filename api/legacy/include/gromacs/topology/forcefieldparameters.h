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

#ifndef GMX_TOPOLOGY_FORCEFIELDPARAMETERS_H
#define GMX_TOPOLOGY_FORCEFIELDPARAMETERS_H

#include <cstdio>

#include <vector>

#include "gromacs/topology/idef.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"

/*! \brief Struct that holds all force field parameters for the simulated system */
struct gmx_ffparams_t
{
    /*! \brief Returns the number of function types, which matches the number of elements in iparams */
    int numTypes() const
    {
        GMX_ASSERT(iparams.size() == functype.size(), "Parameters and function types go together");

        return static_cast<int>(functype.size());
    }

    /* TODO: Consider merging functype and iparams, either by storing
     *       the functype in t_iparams or by putting both in a single class.
     */
    int                     atnr = 0;    /**< The number of non-bonded atom types */
    std::vector<t_functype> functype;    /**< The function type per type */
    std::vector<t_iparams>  iparams;     /**< Force field parameters per type */
    double                  reppow  = 0; /**< The repulsion power for VdW: C12*r^-reppow   */
    real                    fudgeQQ = 0; /**< The scaling factor for Coulomb 1-4: f*q1*q2  */
    gmx_cmap_t              cmap_grid;   /**< The dihedral correction maps                 */
};

void pr_ffparams(FILE* fp, int indent, const char* title, const gmx_ffparams_t* ffparams, gmx_bool bShowNumbers);

#endif
