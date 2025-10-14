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

#ifndef GMX_MDTYPES_NBLIST_H
#define GMX_MDTYPES_NBLIST_H

#include <vector>


struct t_nblist
{
    int              nri    = 0; /* Current number of i particles	   */
    int              maxnri = 0; /* Max number of i particles	   */
    int              nrj    = 0; /* Current number of j particles	   */
    int              maxnrj = 0; /* ;Max number of j particles	   */
    std::vector<int> iinr;       /* The i-elements                        */
    std::vector<int> gid;        /* Index in energy arrays                */
    std::vector<int> shift;      /* Shift vector index                    */
    std::vector<int> jindex;     /* Index in jjnr                         */
    std::vector<int> jjnr;       /* The j-atom list                       */
    std::vector<int> excl_fep;   /* Exclusions for FEP with Verlet scheme */

    int numExclusionsWithinRlist = 0; /* The number of exclusions at distance < rlist */
};

#endif /* GMX_MDTYPES_NBLIST_H */
